"""Run real 2.7B training with recorded Qwen targets and per-rank measurements."""

import atexit
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch

import megatron.training.training as training
import pretrain_rwkv7 as pretrain
from megatron.core import parallel_state as ps
from megatron.core.enums import ModelType
from megatron.training import get_args
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from megatron.training.global_vars import initialize_runtime_services
from rwkv_megatron.arguments import add_rwkv_arguments
from rwkv_megatron.distill_metrics import StudentPerplexityCallback
from rwkv_megatron.online_teacher import TeacherTopK, build_teacher_client

OUT = Path(os.environ["RWKV_EXT_OUTPUT"])
CACHE = Path(os.environ["RWKV_EXT_TEACHER_CACHE"])
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
RECORD = os.environ.get("RWKV_EXT_RECORD") == "1"
ORIGINAL_STEP = training.train_step
ORIGINAL_BATCH = pretrain.get_batch
MEASUREMENTS = []
BATCHES = []
ACTIVE = None
CLIENT = None


class CachedTeacher:
    prompt_mode = "text"

    def __init__(self, args):
        self.original = build_teacher_client(args, use_pool=False)
        self.requests = 0
        self.replays = 0

    def get_effective_vocab_size(self):
        return self.original.get_effective_vocab_size()

    def _get_tokenizer(self):
        return self.original._get_tokenizer()

    def get_logits(self, tokens, **kwargs):
        digest = hashlib.sha256(tokens.detach().cpu().contiguous().numpy().tobytes())
        digest.update(json.dumps(kwargs.get("prompts"), ensure_ascii=False).encode())
        if kwargs.get("attention_mask") is not None:
            digest.update(kwargs["attention_mask"].detach().cpu().numpy().tobytes())
        key = digest.hexdigest()
        path = CACHE / (key + ".pt")
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=True)
            self.replays += 1
        else:
            if not RECORD:
                raise RuntimeError(f"Teacher replay cache missing {key}")
            result = self.original.get_logits(tokens, **kwargs)
            data = dict(
                token_ids=result.token_ids.cpu(), logprobs=result.logprobs.cpu()
            )
            temporary = path.with_suffix(f".rank{torch.distributed.get_rank()}.tmp")
            torch.save(data, temporary)
            temporary.replace(path)
            self.requests += 1
        return TeacherTopK(**data)


def teacher(args):
    global CLIENT
    if CLIENT is None:
        CLIENT = CachedTeacher(args)
    return CLIENT


def batch(*args, **kwargs):
    result = ORIGINAL_BATCH(*args, **kwargs)
    if result[0] is not None:
        tokens, labels, mask = result[:3]
        if ps.get_context_parallel_world_size() > 1:

            def gather(value):
                shards = [
                    torch.empty_like(value)
                    for _ in range(ps.get_context_parallel_world_size())
                ]
                torch.distributed.all_gather(
                    shards, value, group=ps.get_context_parallel_group()
                )
                return torch.cat(shards, dim=1)

            tokens, labels, mask = map(gather, (tokens, labels, mask))
        rows = []
        for token, label, valid in zip(tokens, labels, mask):
            digest = hashlib.sha256()
            for value in (token, label, valid):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            rows.append(
                dict(hash=digest.hexdigest(), valid_tokens=int(valid.sum().item()))
            )
        BATCHES.append(dict(step=ACTIVE, samples=rows))
    return result


def step(*args, **kwargs):
    global ACTIVE
    ACTIVE = kwargs["iteration"] + 1
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = ORIGINAL_STEP(*args, **kwargs)
        torch.cuda.synchronize()
        cfg = get_args()
        MEASUREMENTS.append(
            dict(
                step=ACTIVE,
                seconds=time.perf_counter() - started,
                max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                max_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                losses={key: float(value) for key, value in result[0].items()},
                skipped=int(result[1]),
                grad_norm=float(result[5]),
                global_batch=cfg.global_batch_size,
                sequence_length=cfg.seq_length,
            )
        )
        return result
    finally:
        ACTIVE = None


def finish():
    if not torch.distributed.is_initialized():
        return
    info = dict(
        rank=torch.distributed.get_rank(),
        dp=ps.get_data_parallel_rank(),
        tp=ps.get_tensor_model_parallel_rank(),
        pp=ps.get_pipeline_model_parallel_rank(),
        cp=ps.get_context_parallel_rank(),
        measurements=MEASUREMENTS,
        batches=BATCHES,
        teacher_requests=CLIENT.requests if CLIENT else 0,
        teacher_replays=CLIENT.replays if CLIENT else 0,
    )
    (OUT / f"rank{info['rank']}.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    pretrain._get_online_teacher = teacher
    pretrain.get_batch = batch
    training.train_step = step
    atexit.register(finish)
    with patch(
        "transformers.dynamic_module_utils.resolve_trust_remote_code", return_value=True
    ):
        args = parse_and_validate_args(extra_args_provider=add_rwkv_arguments)
        initialize_runtime_services(args)
        pretrain.train_valid_test_datasets_provider.is_distributed = True
        pretrain.pretrain(
            pretrain_cfg_container_from_args(args),
            pretrain.train_valid_test_datasets_provider,
            ModelType.encoder_or_decoder,
            pretrain.forward_step,
            model_provider=pretrain.model_provider,
            callbacks=[StudentPerplexityCallback()],
        )
