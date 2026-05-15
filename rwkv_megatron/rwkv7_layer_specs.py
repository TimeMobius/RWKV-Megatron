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
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import divide

from einops import einsum, rearrange


from unittest.mock import patch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from megatron.core.utils import is_te_min_version
from megatron.training import get_args
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False
from megatron.core import parallel_state
from megatron.training import print_rank_0

def get_fp8_recipe():
    # 获取同时包含 TP 和 DP 的通信组 (排除 PP)
    # 注意：在较新的 Megatron-Core 版本中，AMAX reduction group 通常由框架维护
    # 如果你是手动调用 te.fp8_autocast，必须显式传入这个组
    
    # 这种组通常称为 "Model Parallel Group" (排除 PP) 或者需要自己构造
    # 在标准 Megatron 中，通常 TE 会复用 tensor_model_parallel_group 
    # 或者你需要一个包含 (TP + DP) 的组。
    
    # 最安全的做法是只在 Tensor Parallel 组内同步 (如果 DP 不需要同步 AMAX)
    # 或者使用 parallel_state.get_tensor_and_data_parallel_group() (如果该函数存在)
    
    fp8_group = parallel_state.get_tensor_model_parallel_group()
    
    return recipe.DelayedScaling(
        margin=0,
        interval=16,
        fp8_format=recipe.Format.HYBRID
        # amax_reduce_group=fp8_group  # <--- 关键修复：指定通信组
    )


def _rwkv7_build_channel_positions(hidden_size: int, device) -> Tensor:
    return torch.arange(hidden_size, device=device, dtype=torch.float32) / hidden_size


def _rwkv7_get_global_layer_id(config: TransformerConfig, layer_number: int) -> int:
    return layer_number + get_transformer_layer_offset(config) - 1


def _rwkv7_time_mix_init_tensors(
    hidden_size: int,
    num_attention_heads: int,
    num_layers: int,
    global_layer_id: int,
    device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    positions = _rwkv7_build_channel_positions(hidden_size, device=device)
    ratio_1_to_almost0 = 1.0 - (global_layer_id / num_layers)

    mix_coeff_input = torch.stack(
        [
            1.0 - torch.pow(positions, 0.2 * ratio_1_to_almost0),
            1.0 - torch.pow(positions, 0.9 * ratio_1_to_almost0),
            1.0 - torch.pow(positions, 0.7 * ratio_1_to_almost0),
            1.0 - torch.pow(positions, 0.7 * ratio_1_to_almost0),
            1.0 - torch.pow(positions, 0.9 * ratio_1_to_almost0),
            1.0 - torch.pow(positions, 0.2 * ratio_1_to_almost0),
        ]
    )

    linear = torch.arange(hidden_size, device=device, dtype=torch.float32) / (hidden_size - 1) - 0.5
    beta = torch.zeros(hidden_size, device=device, dtype=torch.float32) + 0.71 - linear * 0.1
    alpha = torch.zeros(hidden_size, device=device, dtype=torch.float32) + 1.02

    head_dim = divide(hidden_size, num_attention_heads)
    rho = torch.zeros(num_attention_heads, head_dim, device=device, dtype=torch.float32) - 0.04

    return mix_coeff_input, rho.flatten(), beta, alpha


def _rwkv7_channel_mix_init_tensor(
    hidden_size: int,
    num_layers: int,
    global_layer_id: int,
    device,
) -> Tensor:
    positions = _rwkv7_build_channel_positions(hidden_size, device=device)
    ratio_1_to_almost0 = 1.0 - (global_layer_id / num_layers)
    return 1.0 - torch.pow(positions, ratio_1_to_almost0**4)

print(f"[Degugging_Info]:******************************** current have_te is {HAVE_TE}")
with patch("transformers.AutoConfig.register", return_value=None):
    with patch("transformers.AutoModel.register", return_value=None):
        with patch("transformers.AutoModelForCausalLM.register", return_value=None):
            from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
            from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
            from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
            # from fla.modules.token_shift import token_shift
        
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

        self.eps = eps

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
                # setattr(
                #     self.weight_down, "sequence_parallel", self.config.sequence_parallel
                # )

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
            if self.config.sequence_parallel:
                # Flatten tokens only, so TP communication redistributes hidden
                # channels without interleaving different batch samples.
                expanded = rearrange(expanded, "t b d -> (t b) d")
                tp_expanded = all_to_all_sp2hp(expanded)
                tp_expanded = rearrange(
                    tp_expanded,
                    "(t b) d -> t b d",
                    b=x.shape[1],
                )
            else:
                tp_expanded = scatter_to_tensor_model_parallel_region(expanded)
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

        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"bias": 0} if self.to_tensor_parallel and self.use_bias else {},
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

        o = chunk_rwkv7(r, w, k, v, a, b, scale=1., safe_gate=True, chunk_size=64)[0]
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

        args = get_args()
        use_te = args.transformer_impl == "transformer_engine" and HAVE_TE
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
        setattr(self.rho, "allreduce", True)

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

        col_linear_factory = lambda name: (TEColumnParallelLinear if use_te else ColumnParallelLinear)(
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

        group_norm_eps = (
            divide(self.config.hidden_size, self.config.num_attention_heads)
            * self.config.layernorm_epsilon
        )
        self.group_norm = GroupNorm(
            num_groups=self.config.num_attention_heads,
            num_channels=self.config.hidden_size,
            eps=group_norm_eps,
            config=self.config,
        )
        self.linear_proj = (TERowParallelLinear if use_te else RowParallelLinear)(
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
        self.global_layer_id_for_init = _rwkv7_get_global_layer_id(self.config, layer_number)

        self.num_heads_per_partition = divide(
            self.config.num_attention_heads, world_size
        )
        self.head_size = divide(
            self.config.hidden_size, self.config.num_attention_heads
        )

        if self.config.perform_initialization:
            mix_coeff_input, rho_full, beta_full, alpha_full = _rwkv7_time_mix_init_tensors(
                hidden_size=self.config.hidden_size,
                num_attention_heads=self.config.num_attention_heads,
                num_layers=self.config.num_layers,
                global_layer_id=self.global_layer_id_for_init,
                device=self.mix_coeff_input.device,
            )
            shard_size = divide(self.config.hidden_size, world_size)
            shard_start = shard_size * get_tensor_model_parallel_rank()
            shard_end = shard_start + shard_size
            with torch.no_grad():
                self.mix_coeff_input.copy_(mix_coeff_input.to(self.mix_coeff_input.dtype))
                self.rho.copy_(rho_full[shard_start:shard_end].to(self.rho.dtype))
                self.beta.copy_(beta_full[shard_start:shard_end].to(self.beta.dtype))
                self.alpha.copy_(alpha_full[shard_start:shard_end].to(self.alpha.dtype))
    

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

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()  
        pp_rank = get_pipeline_model_parallel_rank()

        v0 = None
        if len(hidden_states.shape) == 4:
            if self.config.sequence_parallel:
                hidden_states, v0 = hidden_states[0], rearrange(hidden_states[1:], 'n t b d -> (n t) b d')
            else:
                hidden_states, v0 = hidden_states[0], hidden_states[1]

        shifted_hidden_states = _token_shift(
            hidden_states, 0, self.config.sequence_parallel
        )
        delta_shift = shifted_hidden_states - hidden_states
        mixed_receptance, mixed_decay, mixed_key, mixed_value, mixed_learn_rate, mixed_gate = fused_addcmul_rwkv7(hidden_states, delta_shift, *[self.mix_coeff_input[i] for i in range(6)])

        learn_rate = torch.sigmoid(self.bottleneck_learn(mixed_learn_rate.contiguous()))
        current_recipe = get_fp8_recipe()
        with te.fp8_autocast(enabled=False, fp8_recipe=current_recipe):
            key = self.linear_key(mixed_key.contiguous())[0]
        rem_key = key * self.beta
        # rem_batch = rem_key.shape[1]
        # rem_seq = rem_key.shape[0]

        temp_rearranged = rearrange(rem_key, 't b (h n) -> b t h n', h=self.num_heads_per_partition)
        normalized_rearranged = torch.nn.functional.normalize(temp_rearranged, dim=-1, p=2.0)
        rem_key_norm = rearrange(normalized_rearranged, 'b t h n -> t b (h n)')

        rep_key = fused_k_rwkv7(key, learn_rate, self.alpha)

        with te.fp8_autocast(enabled=False, fp8_recipe=current_recipe):
            value = self.linear_value(mixed_value.contiguous())[0]

        if v0 is None and self.layer_number == 1 and pp_rank == 0:
            v0 = gather_from_tensor_model_parallel_region(value)

        elif v0 is not None:
            if pp_rank == 0 and self.layer_number == 1: 
                raise ValueError("v0 is not None for the first layer")
            shard_size = divide(self.config.hidden_size, tp_size)
            start_idx = tp_rank * shard_size
            end_idx = start_idx + shard_size
            v0_shard = v0[:, :, start_idx:end_idx]

            vt = torch.sigmoid(self.bottleneck_v(mixed_value.contiguous()))
            # value = value + (v0_shard - value) * vt
            #value = torch.addcmul(value, v0_shard - value, vt)
            value = torch.lerp(value, v0_shard, vt)

        decay = self.bottleneck_decay(mixed_decay.contiguous())
        weight = torch.exp(torch.tensor(-0.5)) * torch.sigmoid(decay)
        with te.fp8_autocast(enabled=False, fp8_recipe=current_recipe):
            receptance = self.linear_receptance(mixed_receptance.contiguous())[0]
        gate = self.bottleneck_gate(mixed_gate.contiguous())

        a = rem_key_norm.neg()
        b = learn_rate * rem_key_norm

        output = self.core_attention(receptance, rep_key, value, weight, a, b)

        output = rearrange(
            self.group_norm(rearrange(output, "t b c -> (t b) c")),
            "(t b) c -> t b c",
            t=output.shape[0],
        )
        # B = output.shape[1]
        # T = output.shape[0]
        H = self.num_heads_per_partition
        # ut = (receptance * (self.rho * rep_key)) * value # torch.Size([4096, 8, 384])
        # ut = ((receptance.view(B,T,H,-1)*rep_key.view(B,T,H,-1)*self.rho.view(1,1,H,-1)).sum(dim=-1, keepdim=True) * value.view(B,T,H,-1)).view(T,B,-1)
        r = rearrange(receptance, 't b (h n) -> b t h n', h=H)
        k = rearrange(rep_key, 't b (h n) -> b t h n', h=H)
        v = rearrange(value, 't b (h n) -> b t h n', h=H)
        #    将 (H*N) -> (1, 1, H, N) 以便广播
        p = rearrange(self.rho, '(h n) -> 1 1 h n', h=H)
        # 2. 以清晰的维度执行计算
        prod = r * k * p
        summed = prod.sum(dim=-1, keepdim=True)
        ut_intermediate = summed * v

        # 3. 使用 rearrange 将结果恢复到目标形状
        #    将 (B, T, H, N) -> (T, B, H*N)
        ut = rearrange(ut_intermediate, 'b t h n -> t b (h n)')
        output += ut
        with te.fp8_autocast(enabled=False, fp8_recipe=current_recipe):
            output, bias = self.linear_proj(output * gate) 

        if self.config.sequence_parallel:
            # reshape v0 to match output shape
            # v0 = v0.reshape(-1, output.shape[0], output.shape[1], output.shape[2])
            v0  = rearrange(v0, '(n t) b d -> n t b d', t=output.shape[0])
            output = torch.cat([output.unsqueeze(0), v0], dim=0)
        else:
            if v0.shape != output.shape:
                raise ValueError(f"v0 shape {v0.shape} does not match output shape {output.shape}")
            output = torch.stack([output, v0], dim=0)
        return output, bias


class RWKV7ChannelMix(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        args = get_args()
        use_te = args.transformer_impl == "transformer_engine" and HAVE_TE
        self.layer_number = None
        self.mix_coeffs = torch.nn.Parameter( 
            torch.empty(config.hidden_size, device=torch.cuda.current_device())
        )
        self.linear_up = (TEColumnParallelLinear if use_te else ColumnParallelLinear)(
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

        self.linear_down = (TERowParallelLinear if use_te else RowParallelLinear)( 
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

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        if self.config.perform_initialization:
            global_layer_id = layer_number - 1
            init_tensor = _rwkv7_channel_mix_init_tensor(
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                global_layer_id=global_layer_id,
                device=self.mix_coeffs.device,
            )
            with torch.no_grad():
                self.mix_coeffs.copy_(init_tensor.to(self.mix_coeffs.dtype))

    def forward(self, hidden_states: Tensor):
        shifted_hidden_states = _token_shift(
            hidden_states, 0, self.config.sequence_parallel
        )
        delta_shift = shifted_hidden_states - hidden_states 
        mixed_up = torch.addcmul(hidden_states, delta_shift, self.mix_coeffs)

        current_recipe = get_fp8_recipe()

        with te.fp8_autocast(enabled=False, fp8_recipe=current_recipe):
            ffn_hidden_states, bias = self.linear_up(mixed_up.contiguous())
        assert bias is None
        ffn_hidden_states = torch.square(torch.relu(ffn_hidden_states))
        with te.fp8_autocast(enabled=False, fp8_recipe=current_recipe):
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

'''
    added by zgz, date:20250619
'''
def get_rwkv7_with_transformer_engine_spec() -> ModuleSpec:
    inp_norm = pre_norm = TENorm if is_te_min_version("1.9.0") else FusedLayerNorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=inp_norm,
            self_attention=ModuleSpec(module=RWKV7TimeMix),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=pre_norm,
            mlp=ModuleSpec(module=RWKV7ChannelMix),
            mlp_bda=get_bias_dropout_add,
        ),

    )
