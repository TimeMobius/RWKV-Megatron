# Modified from Megatron-LM/pretrain_gpt.py
"""
RWKV7预训练脚本

这个脚本实现了RWKV7模型的预训练功能，基于Megatron-LM框架修改。主要功能包括：
1. 模型初始化与加载
2. 数据集处理
3. 分布式训练
4. 损失计算
5. 前向传播步骤定义

主要组件：
- model_provider: 提供RWKV7模型实例
- get_batch: 生成训练批次数据
- loss_func: 定义损失函数
- forward_step: 实现前向传播步骤
- train_valid_test_datasets_provider: 构建训练、验证和测试数据集

Args:
    通过megatron.training.get_args()获取的配置参数

Requirements:
    - megatron-lm
    - rwkv
    - torch
    - transformers
"""
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain RWKV7."""

import os
import math
import rwkv.model
import torch
from functools import partial
import numpy
import torch.distributed

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

import transformers
import rwkv

from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec, get_rwkv7_with_transformer_engine_spec
from rwkv_megatron.rwkv7_model import RWKV7Model


stimer = StragglerDetector()


def model_provider(pre_process=True, post_process=True, config=None, pg_collection=None, vp_stage=None) -> GPTModel:
    """
    RWKV7模型提供者，负责构建和初始化模型实例

    Args:
        pre_process (bool): 是否执行预处理
        post_process (bool): 是否执行后处理

    Returns:
        GPTModel: 初始化好的RWKV7模型实例

    Explanation:
        1. 根据配置参数构建模型配置
        2. 支持从不同来源加载预训练模型：
           - mobius-huggingface
           - rwkv-lm
        3. 包含调试功能，可以打印模型参数信息
    """
    args = get_args()
    if args.context_parallel_size > 1 and args.sequence_parallel:
        raise ValueError(
            "RWKV7 stage-1 context parallelism requires sequence_parallel=false; "
            "combined TP/CP sequence layout is not implemented yet."
        )
    """
    assert (
        args.transformer_impl != "transformer_engine"
    ), "RWKV7 on TransformerEngine not yet implemented"
    """
    use_te = args.transformer_impl == "transformer_engine"
    print_rank_0("building GPT model ...")
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
        # config.token_shift_bottleneck_size = (
        #     args.language_model.token_shift_bottleneck_size
        # )
        # config.decay_bottleneck_size = args.language_model.decay_bottleneck_size
        config.D_AAA_LORA = args.language_model.D_AAA_LORA
        config.D_DECAY_LORA = args.language_model.D_DECAY_LORA
        config.D_GATE_LORA = args.language_model.D_GATE_LORA
        config.D_MV_LORA = args.language_model.D_MV_LORA
    else:
        config = core_transformer_config_from_args(args)
        # config.token_shift_bottleneck_size = args.token_shift_bottleneck_size
        # config.decay_bottleneck_size = args.decay_bottleneck_size
        config.D_AAA_LORA = args.D_AAA_LORA
        config.D_DECAY_LORA = args.D_DECAY_LORA
        config.D_GATE_LORA = args.D_GATE_LORA
        config.D_MV_LORA = args.D_MV_LORA

    config.rwkv_l2wrap_coefficient = getattr(args, "rwkv_l2wrap_coefficient", 0.0)
    config.rwkv_key_fusion = getattr(args, "rwkv_key_fusion", None)
    config.rwkv_fp8_scope = getattr(args, "rwkv_fp8_scope", "ffn")

    assert not getattr(args, "use_legacy_models", False), "RWKV7 only works with Megatron Core"

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        assert args.num_experts is None, "RWKV7 with MoE not yet implemented"
        if use_te:
            print(f"[debugging]: use te ！！！！！")
            transformer_layer_spec = get_rwkv7_with_transformer_engine_spec()
        else:
            transformer_layer_spec = get_rwkv7_layer_spec()

    model = RWKV7Model(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )

    if args.load is None and args.load_pretrained_rwkv is not None:
        print_rank_0(
            f"loading RWKV7 model initialization from {args.load_pretrained_rwkv}"
            f" as '{args.load_pretrained_rwkv_source_type}'"
        )
        if args.load_pretrained_rwkv_source_type == "rwkv-huggingface":
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                args.load_pretrained_rwkv, trust_remote_code=True
            )
            model.from_mobius_huggingface(hf_model, args.check_against_pretrained)
            # raise ValueError(f"RWKV7 model with huggingface not yet implemented")
        elif args.load_pretrained_rwkv_source_type == "rwkv-lm":
            model.from_rwkv_lm(
                torch.load(args.load_pretrained_rwkv, map_location="cpu"), args.check_against_pretrained
            )
        else:
            raise ValueError(f"Unrecognized {args.load_pretrained_rwkv_source_type=}")

    if args.debug_dump:
        print_rank_0(f"vocab_size = {get_tokenizer().vocab_size}")
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                RED = "\033[91m"
                BLUE = "\033[94m"
                RESET = "\033[0m"
                print(f"{RED}============ Parameters @ Rank {i} ============{RESET}")
                for name, param in model.named_parameters():
                    print(f"{BLUE}{name:>80}{RESET}{str(list(param.shape)):>20}")
            torch.distributed.barrier()
        exit(0)

    return model

def model_provider_ckpt(pre_process=True, post_process=True) -> GPTModel:
    """
    RWKV7模型提供者，负责构建和初始化模型实例

    Args:
        pre_process (bool): 是否执行预处理
        post_process (bool): 是否执行后处理

    Returns:
        GPTModel: 初始化好的RWKV7模型实例

    Explanation:
        1. 根据配置参数构建模型配置
        2. 支持从不同来源加载预训练模型：
           - mobius-huggingface
           - rwkv-lm
        3. 包含调试功能，可以打印模型参数信息
    """
    args = get_args()
    assert (
        args.transformer_impl != "transformer_engine"
    ), "RWKV7 on TransformerEngine not yet implemented"

    print_rank_0("building GPT model ...")
    # Experimental loading arguments from yaml

    config = core_transformer_config_from_args(args)
    # config.token_shift_bottleneck_size = args.token_shift_bottleneck_size
    # config.decay_bottleneck_size = args.decay_bottleneck_size
    config.D_AAA_LORA = args.D_AAA_LORA
    config.D_DECAY_LORA = args.D_DECAY_LORA
    config.D_GATE_LORA = args.D_GATE_LORA
    config.D_MV_LORA = args.D_MV_LORA

    config.rwkv_l2wrap_coefficient = getattr(args, "rwkv_l2wrap_coefficient", 0.0)
    config.rwkv_key_fusion = getattr(args, "rwkv_key_fusion", None)

    assert not getattr(args, "use_legacy_models", False), "RWKV7 only works with Megatron Core"

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        assert args.num_experts is None, "RWKV7 with MoE not yet implemented"
        transformer_layer_spec = get_rwkv7_layer_spec()

    model = RWKV7Model(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )

    if args.load is None and args.load_pretrained_rwkv is not None:
        print_rank_0(
            f"loading RWKV7 model initialization from {args.load_pretrained_rwkv}"
            f" as '{args.load_pretrained_rwkv_source_type}'"
        )
        if args.load_pretrained_rwkv_source_type == "mobius-huggingface":
            # hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            #     args.load_pretrained_rwkv, trust_remote_code=True
            # )
            # model.from_mobius_huggingface(hf_model, args.check_against_pretrained)
            raise ValueError(f"RWKV7 model with huggingface not yet implemented")
        elif args.load_pretrained_rwkv_source_type == "rwkv-lm":
            model.from_rwkv_lm(
                torch.load(args.load_pretrained_rwkv, map_location="cpu"), args.check_against_pretrained
            )
        else:
            raise ValueError(f"Unrecognized {args.load_pretrained_rwkv_source_type=}")

    if args.debug_dump:
        print_rank_0(f"vocab_size = {get_tokenizer().vocab_size}")
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                RED = "\033[91m"
                BLUE = "\033[94m"
                RESET = "\033[0m"
                print(f"{RED}============ Parameters @ Rank {i} ============{RESET}")
                for name, param in model.named_parameters():
                    print(f"{BLUE}{name:>80}{RESET}{str(list(param.shape)):>20}")
            torch.distributed.barrier()
        exit(0)

    return model

def get_batch(data_iterator):
    """
    生成训练批次数据

    Args:
        data_iterator: 数据迭代器

    Returns:
        tuple: (tokens, labels, loss_mask, attention_mask, position_ids)

    Explanation:
        1. 处理流水线并行的特殊情况
        2. 根据张量并行rank获取批次数据
        3. 根据上下文并行对批次数据进行切片
    """

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    args = get_args()
    keys = ("tokens", "labels", "loss_mask", "attention_mask", "position_ids")
    batch = {}
    if mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        batch = {key: data[key].cuda(non_blocking=True) if data.get(key) is not None else None for key in keys}
    batch = get_batch_on_this_tp_rank(
        batch,
        has_cu_seqlens=False,
        is_hybrid_cp=False,
        create_attention_mask_in_dataloader=args.create_attention_mask_in_dataloader,
        broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(),
        broadcast_group=mpu.get_tensor_model_parallel_group(),
        cp_size=args.context_parallel_size,
        tp_rank=mpu.get_tensor_model_parallel_rank(),
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        mtp_on_this_rank=False,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        is_pipeline_first_stage=mpu.is_pipeline_first_stage(),
        is_pipeline_last_stage=mpu.is_pipeline_last_stage(),
    )
    batch = get_batch_on_this_cp_rank(
        batch,
        is_hybrid_cp=False,
        cp_group=mpu.get_context_parallel_group(),
        use_contiguous_cp=args.context_parallel_size > 1,
    )
    return [batch[key] for key in keys]


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """
    损失函数计算

    Args:
        loss_mask (torch.Tensor): 用于屏蔽某些位置的损失值
        output_tensor (torch.Tensor): 包含损失值的张量

    Returns:
        tuple: (loss, local_num_tokens, metrics_dict)
            - loss: 当前微批次的损失标量
            - local_num_tokens: 当前微批次中非填充token的数量
            - metrics_dict: 包含损失和token数量等报告指标的字典

    Explanation:
        1. 计算带mask的损失值
        2. 处理上下文并行的情况
        3. 执行损失值的全局归约操作
        4. 支持NaN检查功能
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat(
        [torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)]
    )

    # CP aggregates statistics without placing a c10d collective on the
    # autograd graph. Each rank differentiates its local numerator; DDP
    # averages CP gradients, so the CP multiplier below restores their sum.
    reporting_loss = loss.detach().clone()
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(
            reporting_loss, group=mpu.get_context_parallel_group()
        )
        # Keep the returned forward value as the global sum, with an explicit
        # identity derivative to the local loss (no cross-rank autograd).
        loss = loss + (reporting_loss - loss.detach())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    # reporting_loss already contains the detached global CP statistics.

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {"lm loss": reporting_loss},
    )


def forward_step(data_iterator, model: GPTModel):
    """
    前向传播步骤

    Args:
        data_iterator: 输入数据迭代器
        model (GPTModel): RWKV7模型实例

    Returns:
        tuple: (output_tensor, loss_func)
            - output_tensor: 模型输出张量
            - loss_func: 部分应用的损失函数

    Explanation:
        1. 获取批次数据
        2. 执行模型前向传播
        3. 计时和性能监控
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator
        )
    timers("batch-generator").stop()
    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

# 只在pp的第一个和最后一个stage上构建数据集（并且也要是tp的第一个rank）
def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path),
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        # GPTDatasetConfig的参数
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        shuffle_blend_datasets=args.shuffle_blend_datasets,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """
    构建训练、验证和测试数据集

    Args:
        train_val_test_num_samples: 包含训练、验证和测试样本数量的列表

    Returns:
        tuple: (train_ds, valid_ds, test_ds)
            - train_ds: 训练数据集
            - valid_ds: 验证数据集
            - test_ds: 测试数据集

    Explanation:
        1. 根据参数构建数据集配置
        2. 支持真实数据集和模拟数据集
        3. 使用BlendedMegatronDatasetBuilder构建混合数据集
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def get_train_iters():
    # """Modify the train iters arg from configs according to the dataset length.
    # """
    # args = get_args()
    # auto_train_iters = getattr(args, 'auto_train_iters', None)
    # if auto_train_iters:
    #     if args.mock_data:
    #         dataset_type = MockGPTDataset
    #     else:
    #         dataset_type = GPTDataset
    #     config = core_gpt_dataset_config_from_args(args)
    #     low_level_dataset = dataset_type.build_low_level_dataset(args.data_path[0], config) # total 22847747 sequences
    #     split = config.split_matrix
    #     beg = int(round(split[0][0] * float(low_level_dataset.sequence_lengths.shape[0])))
    #     end = int(round(split[0][1] * float(low_level_dataset.sequence_lengths.shape[0]))) # 22619270 trainning seqences
    #     train_samples =int(numpy.sum(low_level_dataset.sequence_lengths[beg:end]) / args.seq_length)
    #     print_rank_0(f"train_samples is {train_samples}")
    #     args.train_iters = int(train_samples / args.global_batch_size)
    # else:
    #     return
    """
    改进版本：计算能够完整训练所有数据集的训练迭代数
    支持权重模式和无权重模式
    """
    from megatron.training import get_args
    from megatron.training import print_rank_0
    from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset
    
    args = get_args()
    auto_train_iters = getattr(args, 'auto_train_iters', None)
    
    if not auto_train_iters:
        print_rank_0("auto_train_iters is False, using manually set train_iters")
        return
    
    # 判断是否是mock数据
    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    
    # 构建配置（直接使用当前文件中的实现，避免自引用导入）
    config = core_gpt_dataset_config_from_args(args)
    
    print_rank_0("=== 多数据集训练迭代数计算 ===")
    
    total_train_samples = 0
    
    # 解析data_path，使用与get_blend_from_list相同的逻辑
    data_configs = []
    if len(args.data_path) % 2 == 1:
        # 奇数长度：无权重模式 [path1, path2, path3, ...]
        print_rank_0("检测到无权重模式数据配置")
        for path in args.data_path:
            data_configs.append((None, path))
            print_rank_0(f"数据集: {path}")
    else:
        # 偶数长度：可能是权重模式，需要进一步验证
        print_rank_0("检测到偶数长度配置，验证是否为权重模式...")
        
        # 尝试解析为权重模式
        try:
            weights_valid = True
            temp_configs = []
            for i in range(0, len(args.data_path), 2):
                try:
                    weight = float(args.data_path[i])
                    path = args.data_path[i + 1]
                    temp_configs.append((weight, path))
                except (ValueError, TypeError):
                    # 如果无法转换为float，说明不是权重模式
                    weights_valid = False
                    break
            
            if weights_valid:
                # 权重模式：[weight1, path1, weight2, path2, ...]
                print_rank_0("确认为权重模式数据配置")
                data_configs = temp_configs
                for weight, path in data_configs:
                    print_rank_0(f"数据集: {path}, 权重: {weight:,.0f}")
            else:
                # 权重无效，当作无权重模式处理
                print_rank_0("权重解析失败，当作无权重模式处理")
                for path in args.data_path:
                    data_configs.append((None, path))
                    print_rank_0(f"数据集: {path}")
                    
        except Exception as e:
            print_rank_0(f"解析权重时出错: {e}，当作无权重模式处理")
            for path in args.data_path:
                data_configs.append((None, path))
                print_rank_0(f"数据集: {path}")
    
    # 计算每个数据集的训练样本数
    dataset_train_samples = []
    for weight, data_path in data_configs:
        try:
            # 构建底层数据集
            low_level_dataset = dataset_type.build_low_level_dataset(data_path, config)
            
            # 获取分割矩阵
            split = config.split_matrix
            total_sequences = low_level_dataset.sequence_lengths.shape[0]
            
            # 计算训练集范围
            if split[0] is not None:  # 训练集存在
                beg = int(round(split[0][0] * float(total_sequences)))
                end = int(round(split[0][1] * float(total_sequences)))
                
                # 计算训练样本数
                train_tokens = int(numpy.sum(low_level_dataset.sequence_lengths[beg:end]))
                train_samples = train_tokens // args.seq_length
                
                dataset_train_samples.append(train_samples)
                
                print_rank_0(f"数据集 {data_path}:")
                print_rank_0(f"  总序列数: {total_sequences:,}")
                print_rank_0(f"  训练序列范围: [{beg:,}, {end:,})")
                print_rank_0(f"  训练tokens: {train_tokens:,}")
                print_rank_0(f"  训练samples: {train_samples:,}")
                
                if weight is not None:
                    print_rank_0(f"  配置权重: {weight:,.0f}")
            else:
                print_rank_0(f"数据集 {data_path}: 无训练分割")
                dataset_train_samples.append(0)
                
        except Exception as e:
            print_rank_0(f"处理数据集 {data_path} 时出错: {e}")
            dataset_train_samples.append(0)
    
    # 计算总训练样本数
    # 检查是否是权重模式（所有data_configs都有权重）
    has_weights = all(config[0] is not None for config in data_configs)
    
    # if has_weights:
    #     # 权重模式：使用权重加权
    #     weights = [config[0] for config in data_configs]
    #     total_weight = sum(weights)
        
    #     print_rank_0("\n--- 权重模式计算 ---")
    #     for i, ((weight, path), samples) in enumerate(zip(data_configs, dataset_train_samples)):
    #         weight_ratio = weight / total_weight
    #         weighted_samples = int(samples * weight_ratio)
    #         total_train_samples += weighted_samples
    #         print_rank_0(f"数据集 {i+1}: {weighted_samples:,} samples (权重比例: {weight_ratio:.2%})")

    if has_weights:
        # Megatron风格：允许重复以满足权重要求
        weights = [config[0] for config in data_configs]
        total_weight = sum(weights)
        
        # 计算要满足权重比例需要的总样本数
        required_samples = []
        for i, ((weight, path), samples) in enumerate(zip(data_configs, dataset_train_samples)):
            weight_ratio = weight / total_weight
            required = math.ceil(sum(dataset_train_samples) * weight_ratio)
            required_samples.append(required)
            print_rank_0(f"数据集 {i+1}: 实际{samples:,} samples, 权重要求{required:,} samples")
        
        total_train_samples = sum(required_samples)
        print_rank_0(f"总训练样本数（Megatron风格）: {total_train_samples:,}")


    else:
        # 无权重模式：简单求和
        total_train_samples = sum(dataset_train_samples)
        print_rank_0(f"\n--- 无权重模式计算 ---")
        print_rank_0(f"总训练样本数: {total_train_samples:,}")
    
    # 计算训练迭代数
    calculated_train_iters = total_train_samples // args.global_batch_size
    
    print_rank_0(f"\n=== 最终结果 ===")
    print_rank_0(f"总训练样本数: {total_train_samples:,}")
    print_rank_0(f"全局批次大小: {args.global_batch_size}")
    print_rank_0(f"计算得到的训练迭代数: {calculated_train_iters:,}")
    
    # 更新args中的train_iters
    args.train_iters = calculated_train_iters
    print_rank_0(f"已更新 args.train_iters = {calculated_train_iters:,}")

if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    from unittest.mock import patch

    with patch(
        "transformers.dynamic_module_utils.resolve_trust_remote_code",
        return_value=True,
    ):
        from megatron.training.arguments import parse_and_validate_args
        from megatron.training.argument_utils import pretrain_cfg_container_from_args
        from megatron.training.global_vars import initialize_runtime_services
        from rwkv_megatron.arguments import add_rwkv_arguments

        args = parse_and_validate_args(extra_args_provider=add_rwkv_arguments)
        initialize_runtime_services(args)
        get_train_iters()
        full_config = pretrain_cfg_container_from_args(args)
        pretrain(
            full_config,
            train_valid_test_datasets_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            model_provider=model_provider,
        )
