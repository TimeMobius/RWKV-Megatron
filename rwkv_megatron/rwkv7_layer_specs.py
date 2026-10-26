from dataclasses import dataclass, replace
from typing import Optional, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
from einops import einsum, rearrange
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.ops.cp import build_cp_context
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_and_context_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.mappings import (
    all_to_all_hp2sp,
    all_to_all_sp2hp,
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import divide
from megatron.training import get_args

try:
    from megatron.core.extensions.transformer_engine import (
        HAVE_TE,
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
except ImportError:
    HAVE_TE = False

from .key_preparation import prepare_key
from .normalization import get_rwkv7_layernorm_impl
from .precision import keep_time_mix_bf16, validate_rwkv_precision
from .rwkv7_transformer_layer import RWKV7TransformerLayer
from .token_shift import tp_cp_shift_neighbors
from .value_residual import pack_sharded_value, unpack_sharded_value


def mix_inputs(hidden_states, delta, coefficients):
    """Flatten independent tokens into FLA's [batch, time, hidden] layout."""
    shape = hidden_states.shape
    mixed = fused_addcmul_rwkv7(
        hidden_states.reshape(1, -1, shape[-1]),
        delta.reshape(1, -1, shape[-1]),
        *coefficients,
    )
    return tuple(value.reshape(shape) for value in mixed)


def update_key(key, learn_rate, alpha):
    """Keep training tokens out of FLA's one-token decode fallback."""
    shape = key.shape
    return fused_k_rwkv7(
        key.reshape(1, -1, shape[-1]),
        learn_rate.reshape(1, -1, shape[-1]),
        alpha,
    ).reshape(shape)


def _rwkv7_build_channel_positions(hidden_size: int, device) -> Tensor:
    return torch.arange(hidden_size, device=device, dtype=torch.float32) / hidden_size


def _rwkv7_get_global_layer_id(config: TransformerConfig, layer_number: int) -> int:
    return layer_number - 1


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
                setattr(self.bias, "sequence_parallel", self.config.sequence_parallel)

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
        world_group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
        world_rank = get_tensor_model_parallel_rank()
        previous = world_rank - 1 if world_rank > 0 else None
        following = world_rank + 1 if world_rank < world_size - 1 else None
        if get_context_parallel_world_size() > 1:
            world_group = get_tensor_and_context_parallel_group()
            previous, following = tp_cp_shift_neighbors(
                world_group,
                get_context_parallel_rank(),
                world_rank,
                world_size,
                input.device,
            )
        ctx.group = world_group
        ctx.previous = previous
        ctx.following = following
        recv_buffer = torch.empty_like(input.select(dim, 0))
        send_buffer = input.select(dim, -1).contiguous()

        with dist._coalescing_manager(world_group, input.device, async_ops=True) as cm:
            if following is not None:
                world_group.send([send_buffer], following, 0)
            if previous is not None:
                world_group.recv([recv_buffer], previous, 0)

        if previous is None:
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

        world_group = ctx.group
        previous, following = ctx.previous, ctx.following
        recv_buffer = torch.empty_like(grad_output.select(dim, 0))
        send_buffer = grad_output.select(dim, 0).contiguous()

        with dist._coalescing_manager(
            world_group, grad_output.device, async_ops=True
        ) as cm:
            if following is not None:
                world_group.recv([recv_buffer], following, 0)
            if previous is not None:
                world_group.send([send_buffer], previous, 0)

        if following is None:
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


class RWKV7TokenShiftContextParallel(torch.autograd.Function):
    """Token shift across a contiguous CP shard boundary."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input: Tensor, dim: int):
        ctx.dim = dim
        world_group = get_context_parallel_group()
        world_size = get_context_parallel_world_size()
        world_rank = torch.distributed.get_rank(world_group)
        recv_buffer = torch.empty_like(input.select(dim, 0))

        with dist._coalescing_manager(world_group, input.device, async_ops=True) as cm:
            if world_rank < world_size - 1:
                world_group.send(
                    [input.select(dim, -1).contiguous()], world_rank + 1, 0
                )
            if world_rank > 0:
                world_group.recv([recv_buffer], world_rank - 1, 0)

        if world_rank == 0:
            recv_buffer.zero_()
        cm.wait()

        return torch.cat(
            [recv_buffer.unsqueeze(dim), input.narrow(dim, 0, input.size(dim) - 1)],
            dim,
        )

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor):
        dim = ctx.dim
        world_group = get_context_parallel_group()
        world_size = get_context_parallel_world_size()
        world_rank = torch.distributed.get_rank(world_group)
        recv_buffer = torch.empty_like(grad_output.select(dim, 0))

        with dist._coalescing_manager(
            world_group, grad_output.device, async_ops=True
        ) as cm:
            if world_rank < world_size - 1:
                world_group.recv([recv_buffer], world_rank + 1, 0)
            if world_rank > 0:
                world_group.send(
                    [grad_output.select(dim, 0).contiguous()], world_rank - 1, 0
                )

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
        return RWKV7TokenShiftSequenceParallel.apply(input, dim)
    elif get_context_parallel_world_size() > 1:
        return RWKV7TokenShiftContextParallel.apply(input, dim)
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
        w = (
            rearrange(w, "t b (h n) -> b t h n", h=self.num_heads_local)
            .neg()
            .contiguous()
        )
        a = rearrange(a, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()
        b = rearrange(b, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()

        cp_size = get_context_parallel_world_size()
        if cp_size > 1:
            # TP projections/all-to-all restore the complete CP-local sequence
            # when SP is enabled. FLA exchanges state between matching TP heads.
            # Run each sample independently so samples never share a state.
            cp_group = get_context_parallel_group()
            global_seq_len = r.shape[1] * cp_size
            cu_seqlens = torch.tensor(
                [0, global_seq_len], dtype=torch.long, device=r.device
            )
            # Preserve the CP1 BF16 recurrence instead of an affine approximation.
            cp_context = replace(
                build_cp_context(cu_seqlens, group=cp_group, layout="contiguous"),
                dplr_state_mode="recurrent",
            )
            outputs = []
            for batch_idx in range(r.shape[0]):
                output, _ = chunk_rwkv7(
                    r[batch_idx : batch_idx + 1],
                    w[batch_idx : batch_idx + 1],
                    k[batch_idx : batch_idx + 1],
                    v[batch_idx : batch_idx + 1],
                    a[batch_idx : batch_idx + 1],
                    b[batch_idx : batch_idx + 1],
                    scale=1.0,
                    safe_gate=True,
                    chunk_size=64,
                    cp_context=cp_context,
                )
                outputs.append(output)
            o = torch.cat(outputs, dim=0)
        else:
            o = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, safe_gate=True, chunk_size=64)[
                0
            ]
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
        pg_collection=None,
        is_mtp_layer=False,
        name=None,
        pp_layer_offset=None,
        cp_comm_type=None,
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

        # Six independent elementwise coefficient vectors, not a linear weight matrix.
        self.mix_coeff_input.use_muon = False
        setattr(
            self.mix_coeff_input, "sequence_parallel", self.config.sequence_parallel
        )

        validate_rwkv_precision(config, have_te=use_te)
        # Inherit the decoder's Megatron FP8 context, recipe and amax group.
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

    @keep_time_mix_bf16
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
        rotary_pos_cos_sin=None,
        attention_bias=None,
        sequence_len_offset=None,
    ):        

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()  
        pp_rank = get_pipeline_model_parallel_rank()

        v0 = None
        packed_v0 = None
        if len(hidden_states.shape) == 4:
            if self.config.sequence_parallel:
                packed_v0 = hidden_states[1]
                hidden_states, v0 = hidden_states[0], unpack_sharded_value(packed_v0, tp_size)
            else:
                hidden_states, v0 = hidden_states[0], hidden_states[1]

        shifted_hidden_states = _token_shift(
            hidden_states, 0, self.config.sequence_parallel
        )
        delta_shift = shifted_hidden_states - hidden_states
        mixed_receptance, mixed_decay, mixed_key, mixed_value, mixed_learn_rate, mixed_gate = mix_inputs(hidden_states, delta_shift, self.mix_coeff_input.unbind(0))

        learn_rate = torch.sigmoid(self.bottleneck_learn(mixed_learn_rate.contiguous()))
        key = self.linear_key(mixed_key.contiguous())[0]
        rep_key, a, b = prepare_key(key, learn_rate, self.beta, self.alpha, self.head_size, fusion=getattr(self.config, "rwkv_key_fusion", None))

        value = self.linear_value(mixed_value.contiguous())[0]

        if v0 is None and self.layer_number == 1 and pp_rank == 0:
            # SP projections already hold every token for this rank's heads.
            # Keep those heads locally; their gradients flow directly to value.
            v0 = value if self.config.sequence_parallel else gather_from_tensor_model_parallel_region(value)

        elif v0 is not None:
            if pp_rank == 0 and self.layer_number == 1: 
                raise ValueError("v0 is not None for the first layer")
            shard_size = divide(self.config.hidden_size, tp_size)
            start_idx = tp_rank * shard_size
            end_idx = start_idx + shard_size
            v0_shard = v0 if self.config.sequence_parallel else v0[:, :, start_idx:end_idx]

            vt = torch.sigmoid(self.bottleneck_v(mixed_value.contiguous()))
            # value = value + (v0_shard - value) * vt
            #value = torch.addcmul(value, v0_shard - value, vt)
            value = torch.lerp(value, v0_shard, vt)

        decay = self.bottleneck_decay(mixed_decay.contiguous())
        weight = torch.exp(torch.tensor(-0.5)) * torch.sigmoid(decay)
        receptance = self.linear_receptance(mixed_receptance.contiguous())[0]
        gate = self.bottleneck_gate(mixed_gate.contiguous())


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
        output, bias = self.linear_proj(output * gate)

        if self.config.sequence_parallel:
            # Reuse the packed cache in later layers instead of folding it again.
            if packed_v0 is None:
                packed_v0 = pack_sharded_value(v0, tp_size)
            output = torch.stack((output, packed_v0), dim=0)
        else:
            if v0.shape != output.shape:
                raise ValueError(f"v0 shape {v0.shape} does not match output shape {output.shape}")
            output = torch.stack([output, v0], dim=0)
        return output, bias

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Mark TP-sharded local parameters before recursing into child modules."""
        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            {
                "rho": 0,
                "beta": 0,
                "alpha": 0,
            },
            sharded_offsets,
        )
        for name, module in self.named_children():
            sharded_state_dict.update(
                sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata
                )
            )
        return sharded_state_dict


class RWKV7ChannelMix(MegatronModule):
    def __init__(self, config: TransformerConfig, pg_collection=None, is_mtp_layer=False, name=None, hash_moe_layer_threshold=None):
        super().__init__(config)

        args = get_args()
        use_te = args.transformer_impl == "transformer_engine" and HAVE_TE
        validate_rwkv_precision(config, have_te=use_te)
        self.layer_number = None
        self.mix_coeffs = torch.nn.Parameter( 
            torch.empty(config.hidden_size, device=torch.cuda.current_device())
        )
        setattr(self.mix_coeffs, "sequence_parallel", self.config.sequence_parallel)
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

    def forward(self, hidden_states: Tensor, padding_mask=None):
        shifted_hidden_states = _token_shift(
            hidden_states, 0, self.config.sequence_parallel
        )
        delta_shift = shifted_hidden_states - hidden_states 
        mixed_up = torch.addcmul(hidden_states, delta_shift, self.mix_coeffs)


        ffn_hidden_states, bias = self.linear_up(mixed_up.contiguous())
        assert bias is None
        ffn_hidden_states = torch.square(torch.relu(ffn_hidden_states))
        output, bias = self.linear_down(ffn_hidden_states)
        assert bias is None
        return output, None


def get_rwkv7_layer_spec() -> ModuleSpec:
    norm = get_rwkv7_layernorm_impl()
    return ModuleSpec(
        module=RWKV7TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=norm,
            self_attention=ModuleSpec(module=RWKV7TimeMix),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=norm,
            mlp=RWKV7ChannelMix,
            mlp_bda=get_bias_dropout_add,
        ),
    )

'''
    added by zgz, date:20250619
'''
def get_rwkv7_with_transformer_engine_spec() -> ModuleSpec:
    return get_rwkv7_layer_spec()
