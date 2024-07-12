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
            from fla.ops.rwkv6 import fused_recurrent_rwkv6


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


class RWKV6Bottleneck(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        bottleneck_size: int,
        expansion: int,
        to_tensor_parallel: bool,
    ):
        super().__init__(config=config)
        assert not config.use_cpu_initialization

        self.bottleneck_size = bottleneck_size
        self.expansion = expansion

        self.weight_down = torch.nn.Parameter(
            torch.empty(
                self.config.hidden_size,
                expansion,
                bottleneck_size,
                device=torch.cuda.current_device(),
            )
        )
        self.weight_up = torch.nn.Parameter(
            torch.empty(
                expansion,
                bottleneck_size,
                self.config.hidden_size,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.weight_up, "sequence_parallel", self.config.sequence_parallel)
        setattr(self.weight_down, "sequence_parallel", self.config.sequence_parallel)

        self.to_tensor_parallel = to_tensor_parallel
        if self.to_tensor_parallel:
            self.bias = torch.nn.Parameter(
                torch.empty(
                    self.expansion,
                    divide(
                        self.config.hidden_size, get_tensor_model_parallel_world_size()
                    ),
                    device=torch.cuda.current_device(),
                )
            )
            set_tensor_model_parallel_attributes(
                self.bias, is_parallel=True, dim=1, stride=1
            )
            setattr(self.bias, "allreduce", True)
        else:
            self.bias = torch.nn.Parameter(
                torch.empty(
                    self.expansion,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                )
            )
            setattr(
                self.weight_down, "sequence_parallel", self.config.sequence_parallel
            )

        if config.perform_initialization:
            torch.nn.init.uniform_(self.weight_down, -1e-4, 1e-4)
            torch.nn.init.uniform_(self.weight_up, -1e-4, 1e-4)
            torch.nn.init.uniform_(self.bias, -1, 1)

    def forward(self, x: Tensor):
        neck = einsum(x, self.weight_down, "t b d, d e r -> t b e r").tanh()
        expanded = einsum(neck, self.weight_up, "t b e r, e r d -> t b e d")

        if self.to_tensor_parallel:
            expanded = rearrange(expanded, "t b e d -> t (b e d)")
            if self.config.sequence_parallel:
                tp_expanded = all_to_all_sp2hp(expanded)
            else:
                tp_expanded = scatter_to_tensor_model_parallel_region(expanded)
            tp_expanded = rearrange(
                tp_expanded,
                "t (b e d) -> t b e d",
                b=x.shape[1],
                e=self.expansion,
            )
            return tp_expanded + self.bias
        else:
            return expanded + self.bias

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharding along axis 0"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"bias": 1} if self.to_tensor_parallel else {},
            sharded_offsets,
        )


class RWKV6TokenShiftSequenceParallel(torch.autograd.Function):
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
        return RWKV6TokenShiftSequenceParallel.apply(input, 0)
    else:
        return torch.cat(
            [
                torch.zeros_like(input.select(dim, 0)).unsqueeze(dim),
                input.narrow(dim, 0, input.size(dim) - 1),
            ],
            dim,
        )


class WKV6(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ):
        super().__init__(config=config)
        assert not config.use_cpu_initialization

        world_size = get_tensor_model_parallel_world_size()
        self.num_heads_local = divide(self.config.num_attention_heads, world_size)

        self.time_first = torch.nn.Parameter(
            torch.empty(
                divide(self.config.hidden_size, world_size),
                device=torch.cuda.current_device(),
            )
        )

        set_tensor_model_parallel_attributes(
            self.time_first, is_parallel=True, dim=0, stride=1
        )
        setattr(self.time_first, "allreduce", True)

        if self.config.perform_initialization:
            torch.nn.init.normal_(self.time_first, mean=1.0, std=0.02)

    def forward(self, r: Tensor, k: Tensor, v: Tensor, w: Tensor):
        assert r.shape == k.shape == v.shape == w.shape

        r = rearrange(r, "t b (h n) -> b h t n", h=self.num_heads_local).contiguous()
        k = rearrange(k, "t b (h n) -> b h t n", h=self.num_heads_local).contiguous()
        v = rearrange(v, "t b (h n) -> b h t n", h=self.num_heads_local).contiguous()
        w = rearrange(w, "t b (h n) -> b h t n", h=self.num_heads_local).exp().neg().contiguous()
        u = self.time_first.reshape(-1, r.shape[-1]).contiguous()

        o = fused_recurrent_rwkv6(r, k, v, w, u, scale=1.)[0]

        return rearrange(o, "b h t n -> t b (h n)")

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharding along axis 0"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"time_first": 0}, sharded_offsets
        )


class RWKV6TimeMix(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ):
        super().__init__(config=config)

        self.bottleneck_token_shift = RWKV6Bottleneck(
            config,
            config.token_shift_bottleneck_size,
            expansion=5,
            to_tensor_parallel=False,
        )
        self.bottleneck_decay = RWKV6Bottleneck(
            config,
            config.decay_bottleneck_size,
            expansion=1,
            to_tensor_parallel=True,
        )

        self.mix_coeff_input = torch.nn.Parameter(
            torch.empty(
                self.config.hidden_size,
                device=torch.cuda.current_device(),
            )
        )
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
        self.linear_gate = col_linear_factory("gate")

        self.core_attention = WKV6(config, layer_number)

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
    ):
        shifted_hidden_states = _token_shift(
            hidden_states, 0, self.config.sequence_parallel
        )
        delta_shift = shifted_hidden_states - hidden_states
        mixed_input = hidden_states + delta_shift * self.mix_coeff_input

        mix_coeffs = self.bottleneck_token_shift(mixed_input)
        data_dependent_mixed = (
            hidden_states.unsqueeze(-2) + delta_shift.unsqueeze(-2) * mix_coeffs
        )
        mixed_decay, mixed_key, mixed_value, mixed_receptance, mixed_gate = (
            data_dependent_mixed.unbind(-2)
        )

        decay = self.bottleneck_decay(mixed_decay).squeeze(-2)
        key = self.linear_key(mixed_key)[0]
        value = self.linear_value(mixed_value)[0]
        receptance = self.linear_receptance(mixed_receptance)[0]
        gate = torch.nn.functional.silu(self.linear_gate(mixed_gate)[0])

        output = self.core_attention(receptance, key, value, decay)
        output = rearrange(
            self.group_norm(rearrange(output, "t b c -> (t b) c")),
            "(t b) c -> t b c",
            t=output.shape[0],
        )
        output, bias = self.linear_proj(output * gate)

        return output, bias


class RWKV6ChannelMix(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.mix_coeffs = torch.nn.Parameter(
            torch.empty(2, config.hidden_size, device=torch.cuda.current_device())
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
        self.linear_gate = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="cmix_gate",
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
        mixed_input = (
            hidden_states.unsqueeze(-2) + delta_shift.unsqueeze(-2) * self.mix_coeffs
        )
        mixed_up, mixed_gate = mixed_input.unbind(-2)

        ffn_hidden_states, bias = self.linear_up(mixed_up.contiguous())
        assert bias is None
        ffn_hidden_states = torch.square(torch.relu(ffn_hidden_states))
        output, bias = self.linear_down(ffn_hidden_states)
        assert bias is None

        gate, bias = self.linear_gate(mixed_gate.contiguous())
        assert bias is None
        if self.config.sequence_parallel:
            gate = all_to_all_hp2sp(gate)
        else:
            gate = gather_from_tensor_model_parallel_region(gate)
        gate = torch.sigmoid(gate)

        return output * gate.reshape_as(output), None


def get_rwkv6_layer_spec() -> ModuleSpec:
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedLayerNorm,
            self_attention=ModuleSpec(module=RWKV6TimeMix),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=FusedLayerNorm,
            mlp=ModuleSpec(module=RWKV6ChannelMix),
            mlp_bda=get_bias_dropout_add,
        ),
    )
