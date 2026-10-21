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
import time
import rwkv.model
import torch
import torch.nn.functional as F
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
from rwkv_megatron.online_teacher import (
    build_teacher_client, RoundTripDataset, TeacherTopK, TeacherPrefetchIterator, kl_divergence_topk,
)
from rwkv_megatron.jsonl_dataset import JsonlTextDataset


stimer = StragglerDetector()
_online_teacher = None
_teacher_prefetchers = []


def _get_online_teacher(args):
    """Create one HTTP client per training process."""
    global _online_teacher
    if _online_teacher is None:
        _online_teacher = build_teacher_client(args)
        print_rank_0(
            f"online Qwen teacher: {args.rwkv_teacher_model} at {args.rwkv_teacher_url} "
            f"(top_k={args.rwkv_teacher_top_k}, request_temperature=1.0, "
            f"seed={getattr(args, 'rwkv_teacher_seed', None)})"
        )
        if getattr(args, "rwkv_teacher_pool", False):
            print_rank_0(f"shared teacher pool: {args.rwkv_teacher_pool_host}:{args.rwkv_teacher_pool_port}")
    return _online_teacher


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

def get_batch(data_iterator, teacher_prefetch=False):
    """
    生成训练批次数据

    Args:
        data_iterator: 数据迭代器

    Returns:
        tuple: (tokens, labels, loss_mask, attention_mask, position_ids, teacher_future)

    Explanation:
        1. 处理流水线并行的特殊情况
        2. 根据张量并行rank获取批次数据
        3. 根据上下文并行对批次数据进行切片
    """

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None

    # get batches based on the TP rank you are on
    args = get_args()
    if teacher_prefetch:
        from megatron.core.rerun_state_machine import get_rerun_state_machine, RerunMode

        if get_rerun_state_machine().get_mode() != RerunMode.DISABLED:
            raise ValueError("Async teacher prefetch currently requires rerun_mode=disabled")
        if args.dataloader_type != "single" or not hasattr(data_iterator, "iterable"):
            raise ValueError("Async teacher prefetch requires Megatron's single dataloader iterator")
        if not isinstance(data_iterator.iterable, TeacherPrefetchIterator):
            from megatron.core.num_microbatches_calculator import get_num_microbatches

            end_iteration = args.train_iters
            if args.exit_interval:
                end_iteration = min(end_iteration, (args.iteration // args.exit_interval + 1) * args.exit_interval)
            # Constant-batch runs have an exact remaining microbatch count.
            # Avoid requesting unused prompts past a planned short-test exit.
            max_batches = (None if args.rampup_batch_size else
                           max(0, end_iteration - args.iteration) * get_num_microbatches())
            prefetcher = TeacherPrefetchIterator(
                data_iterator.iterable, _get_online_teacher(args),
                num_workers=args.rwkv_teacher_prefetch_workers,
                lookahead=args.rwkv_teacher_prefetch_lookahead,
                max_batches=max_batches,
                min_workers=args.rwkv_teacher_prefetch_min_workers,
                latency_target_seconds=args.rwkv_teacher_prefetch_latency_target,
            )
            # Install below RerunDataIterator; never advance the wrapper ahead
            # of the actual training microbatch. Checkpoint sample accounting
            # stays with Megatron, independent of the prefetch source cursor.
            data_iterator.iterable = prefetcher
            _teacher_prefetchers.append(prefetcher)
            print_rank_0(f"async Qwen teacher: workers={args.rwkv_teacher_prefetch_workers}, "
                         f"lookahead={args.rwkv_teacher_prefetch_lookahead}")
    keys = ("tokens", "labels", "loss_mask", "attention_mask", "position_ids")
    batch = {}
    teacher_future = None
    raw_prompts = None
    if mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        teacher_future = data.get(TeacherPrefetchIterator.future_key)
        raw_prompts = data.get("raw_prompts")
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
    batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=mpu.get_context_parallel_group())
    return [batch[key] for key in keys] + [teacher_future, raw_prompts]


def _finalize_loss(loss: torch.Tensor, token_count: torch.Tensor):
    """Apply Megatron's context-parallel reduction and reporting contract."""
    args = get_args()
    packed = torch.stack([loss * token_count, token_count])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(packed, group=mpu.get_context_parallel_group())
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not packed[0].isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )
    reporting_loss = packed.detach()
    local_num_tokens = packed[1].detach().to(torch.int)
    return packed[0] * args.context_parallel_size, local_num_tokens, {"lm loss": reporting_loss}


def _student_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    valid_vocab_size: int,
) -> torch.Tensor:
    """Return token-normalized student CE over the teacher's effective vocabulary.

    Megatron pads the output head to a TP-friendly vocabulary size.  Those extra
    rows have no tokenizer IDs and must not participate in the CE denominator,
    even when CE is only being collected as a monitoring metric.
    """
    if valid_vocab_size < 1 or valid_vocab_size > logits.shape[-1]:
        raise ValueError(
            f"valid_vocab_size={valid_vocab_size} must be in [1, {logits.shape[-1]}]"
        )
    valid_mask = attention_mask.to(device=logits.device, dtype=torch.bool)
    if valid_mask.any():
        valid_labels = labels[valid_mask]
        if ((valid_labels < 0) | (valid_labels >= valid_vocab_size)).any():
            raise ValueError("student CE labels contain IDs outside the effective vocabulary")
    effective_logits = logits[..., :valid_vocab_size].float()
    token_ce = F.cross_entropy(
        effective_logits.reshape(-1, valid_vocab_size),
        labels.to(device=logits.device, dtype=torch.long).reshape(-1),
        reduction="none",
    ).reshape_as(valid_mask)
    denom = valid_mask.to(token_ce.dtype).sum().clamp_min(1.0)
    return (token_ce * valid_mask.to(token_ce.dtype)).sum() / denom


def distillation_loss_func(loss_mask: torch.Tensor, labels: torch.Tensor, output_tensor, telemetry=None):
    """Compute aligned top-k KL (and optional supervised CE) for one microbatch."""
    args = get_args()
    student_logits, teacher = output_tensor
    if not isinstance(teacher, TeacherTopK):
        raise TypeError(f"Expected TeacherTopK, got {type(teacher)!r}")

    # GPTModel._postprocess returns [batch, sequence, vocab] with labels=None.
    # Use that explicit contract even when batch size equals sequence length.
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"Expected student logits [B,S,V] and labels [B,S], got "
            f"{tuple(student_logits.shape)} and {tuple(labels.shape)}"
        )
    if tuple(student_logits.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            f"Cannot align student logits {tuple(student_logits.shape)} with labels {tuple(labels.shape)}"
        )
    aligned_student = student_logits[:, :-1, :]
    # Megatron loss_mask is already aligned with next-token labels, so mask[t]
    # belongs to student_logits[t] (unlike the source's input attention mask).
    aligned_mask = loss_mask[:, :-1].to(device=student_logits.device, dtype=torch.bool)
    teacher = TeacherTopK(
        token_ids=teacher.token_ids[:, 1:, :],
        logprobs=teacher.logprobs[:, 1:, :],
    )
    valid_vocab_size = getattr(args, "rwkv_distill_valid_vocab_size", None)
    if valid_vocab_size is None:
        # Lightweight unit callers may provide only the loss-related args and
        # already have a synthetic teacher tensor.  The real training parser
        # always provides tokenizer settings, which enables automatic
        # effective-vocabulary discovery.
        if hasattr(args, "rwkv_teacher_tokenizer") or hasattr(args, "tokenizer_model"):
            valid_vocab_size = _get_online_teacher(args).get_effective_vocab_size()
        else:
            valid_vocab_size = student_logits.shape[-1]
    kl = kl_divergence_topk(
        student_logits=aligned_student,
        teacher=teacher,
        attention_mask=aligned_mask,
        temperature=args.rwkv_distill_temperature,
        valid_vocab_size=valid_vocab_size,
    )

    total = float(args.rwkv_distill_logit_weight) * kl
    student_labels = labels[:, :-1].to(device=student_logits.device, dtype=torch.long)
    # Always compute CE for an independent student quality signal.  It only
    # contributes to the training objective when explicitly weighted.
    ce = _student_cross_entropy(aligned_student, student_labels, aligned_mask, valid_vocab_size)
    if args.rwkv_distill_ce_weight:
        total = total + float(args.rwkv_distill_ce_weight) * ce
    token_count = aligned_mask.float().sum()
    loss, count, reporting = _finalize_loss(total, token_count)
    # Reporting pairs follow Megatron's sum/count contract. KL is explicit
    # because the legacy "lm loss" name is the weighted training objective.
    reporting['distill/kl_loss'] = torch.stack((kl.detach() * token_count, token_count))
    reporting['distill/student_ce_loss'] = torch.stack((ce.detach() * token_count, token_count))
    # This is deliberately a separate scalar metric and never enters `total`.
    # Megatron averages scalar metrics over microbatches at the logging interval.
    reporting['distill/student_ppl'] = torch.stack((torch.exp(ce.detach()), torch.ones_like(token_count)))
    if args.rwkv_distill_ce_weight:
        reporting['distill/ce_loss'] = torch.stack((ce.detach() * token_count, token_count))
    for name, value in (telemetry or {}).items():
        reporting[name] = torch.tensor([value, 1.0], device=total.device, dtype=torch.float32)
    return loss, count, reporting


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

    if args.rwkv_distill:
        raise RuntimeError("rwkv_distill requires distillation_loss_func to receive labels")

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat(
        [torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)]
    )

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return loss[0] * args.context_parallel_size, local_num_tokens, {"lm loss": reporting_loss}


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
    if args.rwkv_distill and any(size != 1 for size in (
        args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.context_parallel_size
    )):
        raise ValueError("Initial online distillation path requires tensor/pipeline/context parallel size 1")

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, teacher_future, raw_prompts = get_batch(
            data_iterator,
            teacher_prefetch=(args.rwkv_distill and model.training and args.rwkv_teacher_async_prefetch),
        )
    timers("batch-generator").stop()
    if args.rwkv_distill:
        with stimer:
            output_tensor = model(tokens, position_ids, attention_mask, labels=None)
        # Wait on this exact batch after student forward. Errors propagate;
        # never issue a duplicate synchronous request or skip a sample.
        wait_start = time.perf_counter()
        teacher = (teacher_future.result() if teacher_future is not None
                   else _get_online_teacher(args).get_logits(tokens, prompts=raw_prompts))
        telemetry = {'teacher/wait_seconds': time.perf_counter() - wait_start,
                     'memory/allocated_mib': torch.cuda.memory_allocated() / 2**20,
                     'memory/max_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                     'memory/reserved_mib': torch.cuda.memory_reserved() / 2**20}
        if teacher_future is not None:
            stats = data_iterator.iterable.get_stats()
            telemetry.update({f'teacher/{name}': value for name, value in stats.items()})
        return (output_tensor, teacher), partial(distillation_loss_func, loss_mask, labels, telemetry=telemetry)

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

    jsonl_input = any(
        str(path).endswith(".jsonl") or os.path.isdir(path) for path in args.data_path
    )
    if jsonl_input:
        tokenizer = (_get_online_teacher(args)._get_tokenizer() if args.rwkv_distill else get_tokenizer())
        requested = list(train_val_test_num_samples)
        datasets = []
        for split_name, count in zip(("train", "validation", "test"), requested):
            if not count:
                datasets.append(None)
                continue
            paths = JsonlTextDataset.paths_for_split(args.data_path, split_name)
            dataset = JsonlTextDataset(
                paths, tokenizer, args.seq_length, target_samples=int(count),
                max_records=args.rwkv_jsonl_max_records,
                text_field=args.rwkv_jsonl_text_field,
                chunk_chars=args.rwkv_jsonl_chunk_chars,
            )
            datasets.append(dataset)
            print_rank_0(f"> JSONL {split_name}: {len(dataset)} samples from {len(paths)} files")
        return tuple(datasets)

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    filter_roundtrip = args.rwkv_distill and getattr(args, "rwkv_distill_filter_roundtrip", False)
    requested_samples = list(train_val_test_num_samples)
    build_samples = [n + max(32, n // 20) if n else n for n in requested_samples] if filter_roundtrip else requested_samples
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, build_samples, is_dataset_built_on_rank, config
    ).build()

    if filter_roundtrip:
        import json
        from pathlib import Path

        tokenizer = _get_online_teacher(args)._get_tokenizer()
        filtered = []
        for name, dataset, count in zip(("train", "valid", "test"), (train_ds, valid_ds, test_ds), requested_samples):
            if dataset is not None and count:
                dataset = RoundTripDataset(dataset, tokenizer, count)
                report = {"selected_indices": dataset.indices, "rejected_indices": dataset.rejected_indices,
                          "seed": args.seed, "data_path": args.data_path}
                if args.data_cache_path:
                    target = Path(args.data_cache_path) / f"roundtrip-{name}-selection.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(report, indent=2) + "\n")
                print_rank_0(f"teacher roundtrip filter {name}: kept {len(dataset)}, rejected {dataset.rejected_indices}")
            filtered.append(dataset)
        train_ds, valid_ds, test_ds = filtered

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
        try:
            pretrain(
                full_config,
                train_valid_test_datasets_provider,
                ModelType.encoder_or_decoder,
                forward_step,
                model_provider=model_provider,
            )
        finally:
            for prefetcher in _teacher_prefetchers:
                prefetcher.close()
                print(f"async Qwen teacher final stats: {prefetcher.get_stats()}", flush=True)
