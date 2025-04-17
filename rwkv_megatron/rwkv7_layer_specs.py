from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.distributed as dist

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_context_parallel_world_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.mappings import (
    all_to_all_sp2hp,
    all_to_all_hp2sp,
    scatter_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import divide

from einops import einsum, rearrange


from unittest.mock import patch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

with patch("transformers.AutoConfig.register", return_value=None):
    with patch("transformers.AutoModel.register", return_value=None):
        with patch("transformers.AutoModelForCausalLM.register", return_value=None):
            from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
        
class GroupNorm(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
    ):
        super().__init__(config=config)
        assert not config.use_cpu_initialization

        world_size = get_tensor_model_parallel_world_size()

        self.num_groups = num_groups
        self.num_local_groups = divide(num_groups, world_size)
        self.num_channels = num_channels
        self.num_local_channels = divide(num_channels, world_size)

        self.eps = config.layernorm_epsilon

        self.weight = torch.nn.Parameter(
            torch.empty(self.num_local_channels, device=torch.cuda.current_device())
        )
        self.bias = torch.nn.Parameter(
            torch.empty(self.num_local_channels, device=torch.cuda.current_device())
        )

        set_tensor_model_parallel_attributes(
            self.weight, is_parallel=True, dim=0, stride=1
        )
        setattr(self.weight, "allreduce", True)
        set_tensor_model_parallel_attributes(
            self.bias, is_parallel=True, dim=0, stride=1
        )
        setattr(self.bias, "allreduce", True)

        if config.perform_initialization:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, hidden_states: Tensor):
        return torch.nn.functional.group_norm(
            hidden_states,
            self.num_local_groups,
            self.weight,
            self.bias,
            eps=self.eps,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharding along axis 0"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )


class RWKV7Bottleneck(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        bottleneck_size: int,
        to_tensor_parallel: bool,
        activation_type: str = "tanh",
        use_bias: bool = True
    ):
        super().__init__(config=config)
        assert not config.use_cpu_initialization

        self.bottleneck_size = bottleneck_size
        self.use_bias = use_bias

        self.weight_down = torch.nn.Parameter(
            torch.empty(
                self.config.hidden_size,
                bottleneck_size,
                device=torch.cuda.current_device(),
            )
        )
        self.weight_up = torch.nn.Parameter(
            torch.empty(
                bottleneck_size,
                self.config.hidden_size,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.weight_up, "sequence_parallel", self.config.sequence_parallel)
        setattr(self.weight_down, "sequence_parallel", self.config.sequence_parallel)

        self.to_tensor_parallel = to_tensor_parallel
        if self.use_bias: 
            if self.to_tensor_parallel:
                self.bias = torch.nn.Parameter(
                    torch.empty(
                        divide(
                            self.config.hidden_size, get_tensor_model_parallel_world_size()
                        ),
                        device=torch.cuda.current_device(),
                    )
                )
                set_tensor_model_parallel_attributes(
                    self.bias, is_parallel=True, dim=0, stride=1
                )
                setattr(self.bias, "allreduce", True)
            else:
                self.bias = torch.nn.Parameter(
                    torch.empty(
                        self.config.hidden_size,
                        device=torch.cuda.current_device(),
                    )
                )
                setattr(
                    self.weight_down, "sequence_parallel", self.config.sequence_parallel
                )

            if config.perform_initialization:
                torch.nn.init.uniform_(self.bias, -1, 1)
        else:
            self.bias = None 

        if config.perform_initialization:
            torch.nn.init.uniform_(self.weight_down, -1e-4, 1e-4)
            torch.nn.init.uniform_(self.weight_up, -1e-4, 1e-4)

        if activation_type == "tanh":
            self.activation = torch.tanh
        elif activation_type == "identity":
            self.activation = lambda x: x  
        elif activation_type == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError("Unsupported activation type. Choose from 'tanh', 'identity', or 'sigmoid'.")

    def forward(self, x: Tensor):
        neck = einsum(x, self.weight_down, "t b d, d r -> t b r") 
        neck = self.activation(neck)
        expanded = einsum(neck, self.weight_up, "t b r, r d -> t b d")

        if self.to_tensor_parallel:
            expanded = rearrange(expanded, "t b d -> t (b d)") 
            if self.config.sequence_parallel:
                tp_expanded = all_to_all_sp2hp(expanded)
            else:
                tp_expanded = scatter_to_tensor_model_parallel_region(expanded)
            tp_expanded = rearrange(
                tp_expanded,
                "t (b d) -> t b d",
                b=x.shape[1], # 这里的d实际是d/tp
            )
            return tp_expanded + self.bias if self.use_bias else tp_expanded 
        else:
            return expanded + self.bias if self.use_bias else expanded

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharding along axis 0"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        # import time
        # time.sleep(1000)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"bias": 0} if self.to_tensor_parallel and self.use_bias else {}, # bias特殊，在dim=1上切分
            sharded_offsets,
        )


class RWKV7TokenShiftSequenceParallel(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: Tensor,
        dim: int,
    ):
        ctx.dim = dim

        assert (
            get_context_parallel_world_size() == 1
        ), "Context parallel for RWKV not yet implemented"
        world_group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
        world_rank = get_tensor_model_parallel_rank()

        recv_buffer = torch.empty_like(input.select(dim, 0))

        with dist._coalescing_manager(world_group, input.device, async_ops=True) as cm:
            if world_rank < world_size - 1:
                world_group.send([input.select(dim, -1)], world_rank + 1, 0)
            if world_rank > 0:
                world_group.recv([recv_buffer], world_rank - 1, 0)

        if world_rank == 0:
            recv_buffer.zero_()

        cm.wait()

        return torch.cat(
            [
                recv_buffer.unsqueeze(dim),
                input.narrow(dim, 0, input.size(dim) - 1),
            ],
            dim,
        )

    @staticmethod
    @custom_bwd
    def backward(
        ctx,
        grad_output: Tensor,
    ):
        dim = ctx.dim

        world_group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
        world_rank = get_tensor_model_parallel_rank()

        recv_buffer = torch.empty_like(grad_output.select(dim, 0))

        with dist._coalescing_manager(
            world_group, grad_output.device, async_ops=True
        ) as cm:
            if world_rank < world_size - 1:
                world_group.recv([recv_buffer], world_rank + 1, 0)
            if world_rank > 0:
                world_group.send([grad_output.select(dim, 0)], world_rank - 1, 0)

        if world_rank == world_size - 1:
            recv_buffer.zero_()

        cm.wait()

        return (
            torch.cat(
                [
                    grad_output.narrow(dim, 1, grad_output.size(dim) - 1),
                    recv_buffer.unsqueeze(dim),
                ],
                dim,
            ),
            None,
        )


def _token_shift(input: Tensor, dim: int, sequence_parallel: bool):
    if sequence_parallel:
        return RWKV7TokenShiftSequenceParallel.apply(input, 0)
    else:
        return torch.cat(
            [
                torch.zeros_like(input.select(dim, 0)).unsqueeze(dim),
                input.narrow(dim, 0, input.size(dim) - 1),
            ],
            dim,
        )


class WKV7(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ):
        super().__init__(config=config)
        assert not config.use_cpu_initialization

        world_size = get_tensor_model_parallel_world_size()
        self.num_heads_local = divide(self.config.num_attention_heads, world_size)

    def forward(self, r: Tensor, k: Tensor, v: Tensor, w: Tensor, a: Tensor, b: Tensor):
        assert r.shape == k.shape == v.shape == w.shape

        r = rearrange(r, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()
        k = rearrange(k, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()
        v = rearrange(v, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()
        w = rearrange(w, "t b (h n) -> b t h n", h=self.num_heads_local).neg().contiguous()
        a = rearrange(a, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()
        b = rearrange(b, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()

        o = chunk_rwkv7(r, w, k, v, a, b, scale=1.)[0]
        return rearrange(o, "b t h n -> t b (h n)")

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharding along axis 0"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, sharded_offsets=sharded_offsets
        )

    
class RWKV7TimeMix(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ):
        super().__init__(config=config)

        self.bottleneck_learn = RWKV7Bottleneck(
            config,
            config.D_AAA_LORA,
            to_tensor_parallel=True,
            activation_type="identity"
        )
        self.bottleneck_gate = RWKV7Bottleneck(
            config,
            config.D_GATE_LORA,
            to_tensor_parallel=True,
            activation_type="sigmoid",
            use_bias=False
        )
        self.bottleneck_decay = RWKV7Bottleneck(
            config,
            config.D_DECAY_LORA,
            to_tensor_parallel=True, 
            activation_type="tanh"
        )

        self.bottleneck_v = RWKV7Bottleneck(
            config,
            config.D_MV_LORA,
            to_tensor_parallel=True, 
            activation_type="identity"
        )

        self.mix_coeff_input = torch.nn.Parameter(
            torch.empty(
                6,
                self.config.hidden_size,
                device=torch.cuda.current_device(),
            )
        )

        self.rho = torch.nn.Parameter(
            torch.empty(
                divide(self.config.hidden_size, get_tensor_model_parallel_world_size()),
                device=torch.cuda.current_device(),
            )
        )
        set_tensor_model_parallel_attributes(
            self.rho, is_parallel=True, dim=0, stride=1
        )

        self.beta = torch.nn.Parameter(
            torch.empty(
                divide(self.config.hidden_size, get_tensor_model_parallel_world_size()),
                device=torch.cuda.current_device(),
            )
        )
        set_tensor_model_parallel_attributes(
            self.beta, is_parallel=True, dim=0, stride=1
        )
        setattr(self.beta, "allreduce", True)

        self.alpha = torch.nn.Parameter(
            torch.empty(
                divide(self.config.hidden_size, get_tensor_model_parallel_world_size()),
                device=torch.cuda.current_device(),
            )
        )
        set_tensor_model_parallel_attributes(
            self.alpha, is_parallel=True, dim=0, stride=1
        )
        setattr(self.alpha, "allreduce", True)        

        setattr(
            self.mix_coeff_input, "sequence_parallel", self.config.sequence_parallel
        )

        col_linear_factory = lambda name: ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias, 
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="tmix_" + name,
        )
        self.linear_receptance = col_linear_factory("receptance")
        self.linear_key = col_linear_factory("key")
        self.linear_value = col_linear_factory("value")

        self.core_attention = WKV7(config, layer_number)

        self.group_norm = GroupNorm(
            num_groups=self.config.num_attention_heads,
            num_channels=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            config=self.config,
        )
        self.linear_proj = RowParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="tmix_proj",
        )

        world_size = get_tensor_model_parallel_world_size()
        self.layer_number = layer_number

        self.num_heads_per_partition = divide(
            self.config.num_attention_heads, world_size
        )
        self.head_size = divide(
            self.config.hidden_size, self.config.num_attention_heads
        )
    

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        sequence_len_offset=None,
    ):        
        import os
        pp_group = get_pipeline_model_parallel_group()
        tp_group = get_tensor_model_parallel_group()
        pp_ranks = dist.get_process_group_ranks(pp_group)
        tp_ranks = dist.get_process_group_ranks(tp_group)
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()
        pp_rank = get_pipeline_model_parallel_rank()

        current_rank = dist.get_rank()
        v0 = None  # 初始化 v0
        if len(hidden_states.shape) == 4 and hidden_states.shape[0] == 2:
            hidden_states, v0 = hidden_states[0], hidden_states[1]

        shifted_hidden_states = _token_shift( 
            hidden_states, 0, self.config.sequence_parallel
        )
        delta_shift = shifted_hidden_states - hidden_states
        mixed_input = hidden_states.unsqueeze(-2) + delta_shift.unsqueeze(-2) * self.mix_coeff_input
        mixed_receptance, mixed_key, mixed_value, mixed_decay, mixed_learn_rate, mixed_gate = mixed_input.unbind(-2)
        learn_rate = torch.sigmoid(self.bottleneck_learn(mixed_learn_rate.contiguous())) 
        key = self.linear_key(mixed_key.contiguous())[0]
        rem_key = key * self.beta

        rem_key_norm = torch.nn.functional.normalize(rem_key.view(rem_key.shape[0],rem_key.shape[1],self.num_heads_per_partition, self.head_size), dim=-1, p=2.0) # κ^t TODO sequence_parallel 适配 1204 -> 4096
        
        rep_key = key * (1 + (learn_rate - 1) * self.alpha)
        value = self.linear_value(mixed_value.contiguous())[0] 

        if v0 is None and self.layer_number == 1 and pp_rank == 0:
            if self.config.sequence_parallel:
                v0 = all_to_all_hp2sp(value) # TODO 需要修改
            else:
                v0 = gather_from_tensor_model_parallel_region(value)

        elif v0 is not None:
            if pp_rank == 0 and self.layer_number == 1: 
                raise ValueError("v0 is not None for the first layer")
            shard_size = divide(self.config.hidden_size, tp_size)
            start_idx = tp_rank * shard_size
            end_idx = start_idx + shard_size
            v0_shard = v0[:, :, start_idx:end_idx]


            vt = torch.sigmoid(self.bottleneck_v(mixed_value.contiguous()))
            value = v0_shard + (value - v0_shard) * vt

        decay = self.bottleneck_decay(mixed_decay.contiguous())
        weight = torch.exp(torch.tensor(-0.5)) * torch.sigmoid(decay)
        receptance = self.linear_receptance(mixed_receptance.contiguous())[0]
        gate = self.bottleneck_learn(mixed_gate.contiguous())
        rem_key_norm = rearrange(rem_key_norm, "t b n d -> t b (n d)")
        a = rem_key_norm.neg()
        b = learn_rate * rem_key_norm

        output = self.core_attention(receptance, rep_key, value, weight, a, b)

        output = rearrange(
            self.group_norm(rearrange(output, "t b c -> (t b) c")),
            "(t b) c -> t b c",
            t=output.shape[0],
        )
        ut = (receptance * (self.rho * rep_key)) * value
        output += ut
        output, bias = self.linear_proj(output * gate)

        if v0.shape != output.shape:
            raise ValueError(f"v0 shape {v0.shape} does not match output shape {output.shape}")
        output = torch.stack([output, v0], dim=0)
        return output, bias


class RWKV7ChannelMix(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.mix_coeffs = torch.nn.Parameter( 
            torch.empty(config.hidden_size, device=torch.cuda.current_device())
        )
        self.linear_up = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="cmix_up",
        )

        self.linear_down = RowParallelLinear( 
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="cmix_down",
        )

    def forward(self, hidden_states: Tensor):
        shifted_hidden_states = _token_shift(
            hidden_states, 0, self.config.sequence_parallel
        )
        delta_shift = shifted_hidden_states - hidden_states 
        mixed_up = (
            hidden_states + delta_shift * self.mix_coeffs
        )

        ffn_hidden_states, bias = self.linear_up(mixed_up.contiguous())
        assert bias is None
        ffn_hidden_states = torch.square(torch.relu(ffn_hidden_states))
        output, bias = self.linear_down(ffn_hidden_states)
        assert bias is None
        return output, None


def get_rwkv7_layer_spec() -> ModuleSpec:
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm,
            self_attention=ModuleSpec(module=RWKV7TimeMix),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=ModuleSpec(module=RWKV7ChannelMix),
            mlp_bda=get_bias_dropout_add,
        ),
    )
