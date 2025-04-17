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

    def sharded_state_dict( # 返回分片状态字典，用于分布式检查点保存
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

# 这个bottleneck指的是ddlerp中的lora部分，不包含time shift
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
        if self.use_bias:  # 只有在 use_bias 为 True 时才创建 bias 参数
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
            self.bias = None  # 如果不使用 bias，将其设置为 None

        if config.perform_initialization:
            torch.nn.init.uniform_(self.weight_down, -1e-4, 1e-4)
            torch.nn.init.uniform_(self.weight_up, -1e-4, 1e-4)
            # torch.nn.init.uniform_(self.bias, -1, 1)

        # 根据 activation_type 设置激活函数
        if activation_type == "tanh":
            self.activation = torch.tanh
        elif activation_type == "identity":
            self.activation = lambda x: x  # Identity 函数，直接返回输入
        elif activation_type == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError("Unsupported activation type. Choose from 'tanh', 'identity', or 'sigmoid'.")

    def forward(self, x: Tensor): # 输入 x：一个形状为 [t, b, d] 的张量, 其中 t 是序列长度, b 是批量大小, d 是隐藏维度
        neck = einsum(x, self.weight_down, "t b d, d r -> t b r") # d e r 指的是hidden_size, expansion (5), bottleneck_size (32)，这里就是把d维度降维到r维度，有e个r
        neck = self.activation(neck)  # 使用存储的激活函数
        expanded = einsum(neck, self.weight_up, "t b r, r d -> t b d") # 把e个r维度扩展到d维度

        if self.to_tensor_parallel:
            expanded = rearrange(expanded, "t b d -> t (b d)") 
            if self.config.sequence_parallel:
                tp_expanded = all_to_all_sp2hp(expanded) # 这个函数不仅合并了sp，而且同时处理了hp的张量分布
            else:
                tp_expanded = scatter_to_tensor_model_parallel_region(expanded) # 这里的expanded的d是完整的d，需要进行tp切分
            tp_expanded = rearrange(
                tp_expanded,
                "t (b d) -> t b d",
                b=x.shape[1], # 这里的d实际是d/tp
            )
            return tp_expanded + self.bias if self.use_bias else tp_expanded # 这里bias也在init的时候切分了
        else:
            return expanded + self.bias if self.use_bias else expanded # self.bias shape (,d)

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
            {"bias": 1} if self.to_tensor_parallel and self.use_bias else {}, # bias特殊，在dim=1上切分
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

    # 这个函数计算的是一个tp rank下的wkv，而不是全局，所以u的大小为 divide(self.config.hidden_size, world_size)，即 h/d，但是使用的时候self.core_attention(receptance, key, value, decay)传入的
    # 完整的张量，所以是在forward调用之前被megatron内部函数进行了TP的拆分
    def forward(self, r: Tensor, k: Tensor, v: Tensor, w: Tensor, a: Tensor, b: Tensor):
        assert r.shape == k.shape == v.shape == w.shape

        r = rearrange(r, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()
        k = rearrange(k, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()
        v = rearrange(v, "t b (h n) -> b t h n", h=self.num_heads_local).contiguous()
        w = rearrange(w, "t b (h n) -> b t h n", h=self.num_heads_local).neg().contiguous() # 没有第二个exp，因为fla要求w是log space
        # u = self.time_first.reshape(-1, r.shape[-1]).contiguous() # 这里的u是直接通过加载权重得到的，没有做任何并行策略, u重组成[head num, dim of head]
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

_GLOBAL_V0_VALUE = None

class RWKV7TimeMixV0:
    @staticmethod
    def set_value_init(value):
        """设置进程内共享的v0值，保留梯度信息"""
        global _GLOBAL_V0_VALUE
        _GLOBAL_V0_VALUE = value
        
    @staticmethod
    def get_value_init():
        """获取进程内共享的v0值"""
        global _GLOBAL_V0_INIT_FLAG
        return _GLOBAL_V0_VALUE
    
    
class RWKV7TimeMix(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
    ):
        super().__init__(config=config)

        self.bottleneck_learn = RWKV7Bottleneck( # at的lora部分
            config,
            config.D_AAA_LORA,
            to_tensor_parallel=True,
            activation_type="identity"
        )
        self.bottleneck_gate = RWKV7Bottleneck(  # g的lora部分
            config,
            config.D_GATE_LORA,
            to_tensor_parallel=True,
            activation_type="sigmoid",
            use_bias=False
        )
        self.bottleneck_decay = RWKV7Bottleneck( # d的lora部分
            config,
            config.D_DECAY_LORA,
            to_tensor_parallel=True, 
            activation_type="tanh"
        )

        self.bottleneck_v = RWKV7Bottleneck( # v的lora部分，只有非0层有
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

        # TODO 加self可能是切分策略下的维度，不加self就是原始config文件里的参数，并不是，这两个一样
        # self.rho = torch.nn.Parameter(
        #     torch.empty(
        #         config.num_attention_heads,
        #         divide(config.hidden_size, config.num_attention_heads),
        #         device=torch.cuda.current_device(),
        #     )
        # )
        self.rho = torch.nn.Parameter(
            torch.empty(
                divide(self.config.hidden_size, get_tensor_model_parallel_world_size()),
                device=torch.cuda.current_device(),
            )
        )
        # 设置张量并行属性
        set_tensor_model_parallel_attributes(
            self.rho, is_parallel=True, dim=0, stride=1
        )

        self.beta = torch.nn.Parameter(
            torch.empty(
                divide(self.config.hidden_size, get_tensor_model_parallel_world_size()),
                device=torch.cuda.current_device(),
            )
        )
        # 设置张量并行属性
        set_tensor_model_parallel_attributes(
            self.beta, is_parallel=True, dim=0, stride=1
        )
        # 设置是否需要在反向传播时进行allreduce操作
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

        # 注册为可训练参数，用于存储第一层的value初始值
        # 这个参数将通过同步机制在所有进程间共享
        self.v0 = torch.nn.Parameter(
            torch.zeros(
                divide(self.config.hidden_size, get_tensor_model_parallel_world_size()),
                device=torch.cuda.current_device(),
            )
        )
        # 设置张量并行属性
        set_tensor_model_parallel_attributes(
            self.v0, is_parallel=True, dim=0, stride=1
        )
        setattr(self.v0, "allreduce", True)
        
        # 为所有批次和序列的v0值创建缓存
        # 形状为 [batch_size, seq_len, hidden_dim/tp_world_size]
        self.register_buffer(
            "v0_cached", 
            torch.zeros(
                1,  # 初始化为1，会在实际使用时调整大小
                1,  # 初始化为1，会在实际使用时调整大小
                divide(self.config.hidden_size, get_tensor_model_parallel_world_size()),
                device=torch.cuda.current_device(),
            )
        )
        # 设置张量并行属性
        set_tensor_model_parallel_attributes(
            self.v0_cached, is_parallel=True, dim=2, stride=1
        )
        
        # 添加标志，表示v0_cached是否已经初始化
        self.register_buffer(
            "v0_initialized",
            torch.tensor(0, dtype=torch.bool, device=torch.cuda.current_device())
        )

        setattr(
            self.mix_coeff_input, "sequence_parallel", self.config.sequence_parallel
        )

        # setattr(
        #     self.beta, "sequence_parallel", self.config.sequence_parallel
        # )

        # setattr(
        #     self.alpha, "sequence_parallel", self.config.sequence_parallel
        # )

        col_linear_factory = lambda name: ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias, # add_bias_linear = False  add_qkv_bias: False
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
        # self.pp_rank = get_pipeline_model_parallel_rank()
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
        shifted_hidden_states = _token_shift(  # b - a
            hidden_states, 0, self.config.sequence_parallel
        )
        delta_shift = shifted_hidden_states - hidden_states
        mixed_input = hidden_states.unsqueeze(-2) + delta_shift.unsqueeze(-2) * self.mix_coeff_input # mixed_input（[t, b, d]）
        mixed_receptance, mixed_key, mixed_value, mixed_decay, mixed_learn_rate, mixed_gate = mixed_input.unbind(-2)
        learn_rate = torch.sigmoid(self.bottleneck_learn(mixed_learn_rate.contiguous())) # at
        key = self.linear_key(mixed_key.contiguous())[0] # kt
        rem_key = key * self.beta # κt
        # from time import sleep
        # sleep(2000)
        rem_key_norm = torch.nn.functional.normalize(rem_key.view(hidden_states.shape[0],hidden_states.shape[1],self.num_heads_per_partition, self.head_size), dim=-1, p=2.0) # κ^t TODO sequence_parallel 适配 1204 -> 4096
        rep_key = key * (1 + (learn_rate - 1) * self.alpha) # k~t
        value = self.linear_value(mixed_value.contiguous())[0] # shape is [t, b, d] d 是tp切分的384

        world_group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pp_group = get_pipeline_model_parallel_group()
        pp_size = get_pipeline_model_parallel_world_size()
        pp_rank = get_pipeline_model_parallel_rank()

#         import os
#         print(f"""
# 进程信息:
# - 进程ID (PID): {os.getpid()}
# - 张量并行组: {world_group}
# - 张量并行大小: {world_size}
# - 张量并行排名: {world_rank}
# - 流水线并行组: {pp_group}
# - 流水线并行大小: {pp_size}
# - 流水线并行排名: {pp_rank}
# - layer_number: {self.layer_number}

# 模型配置:
# - self.config.hidden_size: {self.config.hidden_size}
# - self.config.num_attention_heads: {self.config.num_attention_heads}
# - self.head_size: {self.head_size}
# - self.mix_coeff_input shape: {self.mix_coeff_input.shape}
# - linear_receptance weight shape: {self.linear_receptance.weight.shape}")
#             """)

       #  from megatron.core import mpu
        import os
        pp_group = get_pipeline_model_parallel_group()  # 获取正确的 PP group
        tp_group = get_tensor_model_parallel_group()
        pp_ranks = dist.get_process_group_ranks(pp_group)
        tp_ranks = dist.get_process_group_ranks(tp_group)
        current_rank = dist.get_rank()
        # pp_rank是global pp rank (stage) ID
        # tp_rank是当前stage 的tp rank ID
        # dist.get_rank() 是global rank 排名，即全局卡的ID

        # print(f"pp_rank {pp_rank} tp rank {tp_rank} Rank {dist.get_rank()}: pp_group ranks = {pp_ranks}, tp_group ranks = {tp_ranks}, layer_number = {self.layer_number}")
        # pp_rank 0 tp rank 0 Rank 0: pp_group ranks = [0, 2, 4, 6], tp_group ranks = [0, 1]
        # pp_rank 0 tp rank 1 Rank 1: pp_group ranks = [1, 3, 5, 7], tp_group ranks = [0, 1]

        # 每个出现3遍

        # 处理v0：如果是第一层，计算和存储v0；如果是后续层，应用v0 transformer
        # if 是第一个 pp rank = 0 并且是第一层
        # 将v0进行broadcast，，让其他进程可以访问
        # 否则如果是其他rank和其他层，则先等待vobroadcast，如果已经broadcast，则使用v0，否则等待
        # v0 = RWKV7TimeMixV0.get_value_init()
        print(f"pp_rank {pp_rank} tp rank {tp_rank} Rank {dist.get_rank()}: pp_group ranks = {pp_ranks}, tp_group ranks = {tp_ranks}, layer_number = {self.layer_number}, 进程ID (PID): {os.getpid()}")
        
        if pp_rank == 0 and self.layer_number == 1:
            # 在第一层初始化v0_cached，存储所有批次和序列的value
            v0 = value.clone()  # 保留梯度连接
            RWKV7TimeMixV0.set_value_init(v0)
            # 广播到其他流水线阶段
            with torch.no_grad():
                work_handle = dist.broadcast(v0, src=dist.get_rank(), group=pp_group, async_op=True)
                work_handle.wait()
        elif pp_rank != 0 and self.layer_number == 1:
            v0 = torch.zeros_like(value, requires_grad=False)
            with torch.no_grad():
                work_handle = dist.broadcast(v0, src=dist.get_rank(), group=pp_group, async_op=True)
                work_handle.wait()
                
            RWKV7TimeMixV0.set_value_init(v0)
            vt = torch.sigmoid(self.bottleneck_v(mixed_value.contiguous()))
            value = v0 + (value - v0) * vt
        else: # self.layer_number != 1
            v0 = RWKV7TimeMixV0.get_value_init()
            vt = torch.sigmoid(self.bottleneck_v(mixed_value.contiguous()))
            value = v0 + (value - v0) * vt










        decay = self.bottleneck_decay(mixed_decay.contiguous()) # 去掉expanded维度，得到[t, b, d]
        weight = torch.exp(torch.tensor(-0.5)) * torch.sigmoid(decay)
        receptance = self.linear_receptance(mixed_receptance.contiguous())[0]
        gate = self.bottleneck_learn(mixed_gate.contiguous())
        rem_key_norm = rearrange(rem_key_norm, "t b n d -> t b (n d)")
        a = rem_key_norm.neg()
        b = learn_rate * rem_key_norm

        output = self.core_attention(receptance, rep_key, value, weight, a, b) # ab代表什么 结果是rwkv的输出

        output = rearrange(
            self.group_norm(rearrange(output, "t b c -> (t b) c")),
            "(t b) c -> t b c",
            t=output.shape[0],
        )
        ut = (receptance * (self.rho * rep_key)) * value
        output += ut
        output, bias = self.linear_proj(output * gate)

        return output, bias


class RWKV7ChannelMix(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.mix_coeffs = torch.nn.Parameter( # 这里的参数指的是ffn里面的用于lerp的系数，即x_k，可训练的
            torch.empty(config.hidden_size, device=torch.cuda.current_device())
        )
        self.linear_up = ColumnParallelLinear( # 上投影层，Wk'
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
        # self.linear_gate = ColumnParallelLinear( # 门控投影，Wr ′
        #     self.config.hidden_size,
        #     self.config.hidden_size,
        #     config=self.config,
        #     init_method=self.config.init_method,
        #     gather_output=False,
        #     bias=self.config.add_bias_linear,
        #     skip_bias_add=False,
        #     is_expert=False,
        #     tp_comm_buffer_name="cmix_gate",
        # )
        self.linear_down = RowParallelLinear( # 下投影层 Wv′
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
        # print(hidden_states.shape) # 应该是[t,b,c]
        print("555555555555555555555555")
        shifted_hidden_states = _token_shift(
            hidden_states, 0, self.config.sequence_parallel # 默认sequence_parallel = True
        )
        delta_shift = shifted_hidden_states - hidden_states # xt-1 - xt
        mixed_up = (
            hidden_states + delta_shift * self.mix_coeffs # mix——coeffs是逐token（在hidden维度的系数），在seq和batch上都是1维度，需要广播，即每个token都是一样的系数
        )

        ffn_hidden_states, bias = self.linear_up(mixed_up.contiguous())
        assert bias is None
        ffn_hidden_states = torch.square(torch.relu(ffn_hidden_states))
        output, bias = self.linear_down(ffn_hidden_states)
        assert bias is None
        # 是否还需要进行all to all
        # gate, bias = self.linear_gate(mixed_gate.contiguous())
        # assert bias is None
        # if self.config.sequence_parallel:
        #     gate = all_to_all_hp2sp(gate)
        # else:
        #     gate = gather_from_tensor_model_parallel_region(gate)
        # gate = torch.sigmoid(gate)

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
