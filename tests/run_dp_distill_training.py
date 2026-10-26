"""DP audit through production DDP/schedules/Adam/checkpoint, deterministic teacher.

RWKV_DP_AUDIT_DIR selects snapshots; RWKV_DP_FP32=1 uses a torch recurrence
oracle. Inputs are indexed by Megatron's real data sampler, never by rank.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

import megatron.training.training as training
import pretrain_rwkv7 as pretrain
from megatron.core import parallel_state as ps
from megatron.core.enums import ModelType
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from megatron.training.global_vars import initialize_runtime_services
from rwkv_megatron.arguments import add_rwkv_arguments
from rwkv_megatron.distill_metrics import StudentPerplexityCallback
from rwkv_megatron.online_teacher import TeacherTopK

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training-smoke/audit"))
from distributed import recurrence

OUT = Path(os.environ["RWKV_DP_AUDIT_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
active = None


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def optimizer_snapshot(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        return [optimizer_snapshot(child) for child in children]
    inner = optimizer.optimizer
    actual = getattr(inner, "sub_optimizers", [inner])
    residency = {}
    states = []
    for child in actual:
        # Inspect live FP32 state directly; TE's exporter rejects the generic
        # loader's unused per-parameter step before it can be filtered here.
        state = cpu_snapshot(torch.optim.Optimizer.state_dict(child))
        # FusedAdam uses group['step']; checkpoint loading can leave unused
        # per-parameter 'step' entries supplied by the generic shard loader.
        if type(child).__name__ == "FusedAdam":
            assert all("step" in group for group in child.param_groups)
            for values in state["state"].values():
                values.pop("step", None)
        elif isinstance(child, torch.optim.Adam):
            # torch Adam advances state[param]['step'], never group['step'].
            assert all("step" in values for values in child.state.values())
            for group in state["param_groups"]:
                group.pop("step", None)
        states.append(state)
        for values in child.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    device = value.device.type
                    residency[device] = residency.get(device, 0) + value.numel()
    return dict(
        state=states,
        residency=residency,
        inner_params=(
            cpu_snapshot(
                [
                    [param for group in child.param_groups for param in group["params"]]
                    for child in actual
                ]
            )
            if hasattr(inner, "sub_optimizers")
            else []
        ),
        masters=cpu_snapshot(
            getattr(
                optimizer,
                "shard_fp32_from_float16_groups",
                getattr(optimizer, "fp32_from_float16_groups", []),
            )
        ),
    )


def gradient_snapshot(module, optimizer):
    children = getattr(optimizer, "chained_optimizers", [optimizer])
    distributed = [child for child in children if hasattr(child, "gbuf_ranges")]
    result = {}
    for key, param in module.named_parameters():
        grad = param.main_grad.detach()
        if distributed:
            # Reduce-scatter leaves only the owned slice reduced. Other entries
            # of main_grad are stale local values, not replicated gradients.
            full = torch.zeros_like(grad)
            for child in distributed:
                if param in child.model_param_gbuf_map:
                    owned = child._get_model_param_range_map(param)["param"]
                    full.view(-1)[owned.start : owned.end].copy_(
                        grad.view(-1)[owned.start : owned.end]
                    )
            torch.distributed.all_reduce(full, group=distributed[0].data_parallel_group)
            grad = full
        result[key] = grad.cpu().clone()
    return result


class FixtureDataset(Dataset):
    def __init__(self, count, split):
        self.count, self.split = count, split

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(20261023 + self.split * 100000 + idx)
        tokens = torch.randint(1, 119, (129,), generator=generator)
        mask = (torch.arange(128) < 128 - (idx % 5) * 9).float()
        return dict(
            tokens=tokens[:-1],
            labels=tokens[1:].masked_fill(~mask.bool(), -100),
            loss_mask=mask,
            position_ids=torch.arange(128),
        )


def datasets(counts):
    if not pretrain.is_dataset_built_on_rank():
        return None, None, None
    return tuple(
        FixtureDataset(int(n), split) if n else None for split, n in enumerate(counts)
    )


datasets.is_distributed = True


class FixtureTeacher:
    prompt_mode = "text"

    def get_logits(self, tokens, **kwargs):
        if ps.get_context_parallel_world_size() > 1:
            assert ps.get_context_parallel_rank() == 0
            assert tokens.shape[1] == 128, "Teacher must receive full context"
        if active is not None:
            active["teacher_requests"] += 1
        if os.environ.get("RWKV_DP_FAIL") == "1" and ps.get_data_parallel_rank() == 1:
            raise TimeoutError("injected DP1 teacher failure")
        ids = (
            tokens[..., None] + torch.tensor([0, 31, 67, 98], device=tokens.device)
        ) % 119
        lp = (
            -torch.tensor([0.2, 1.1, 2.3, 4.0], device=tokens.device)
            .expand_as(ids)
            .clone()
        )
        # Absent teacher rows depend on sample content, never on DP/TP rank.
        ids = ids.masked_fill((tokens % 23 == 0)[..., None], -1)
        return TeacherTopK(ids, lp)


def unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def identity(tokens, labels, mask):
    digest = hashlib.sha256()
    for value in (tokens, labels, mask):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


original_batch = pretrain.get_batch
original_forward = pretrain.forward_step
original_step = training.train_step


def audited_batch(*args, **kwargs):
    result = original_batch(*args, **kwargs)
    if active is not None and result[0] is not None:
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
        active["current_samples"] = [
            identity(t, l, m) for t, l, m in zip(tokens, labels, mask)
        ]
        active["samples"].extend(active["current_samples"])
    return result


def audited_forward(iterator, model):
    output, loss = original_forward(iterator, model)
    if active is not None and ps.is_pipeline_last_stage():
        logits = output[0].detach()
        shards = [
            torch.empty_like(logits)
            for _ in range(ps.get_tensor_model_parallel_world_size())
        ]
        torch.distributed.all_gather(
            shards, logits, group=ps.get_tensor_model_parallel_group()
        )
        full = torch.cat(shards, -1)
        if ps.get_context_parallel_world_size() > 1:
            contexts = [
                torch.empty_like(full)
                for _ in range(ps.get_context_parallel_world_size())
            ]
            torch.distributed.all_gather(
                contexts, full, group=ps.get_context_parallel_group()
            )
            full = torch.cat(contexts, dim=1)
        full = full.cpu()
        for key, value in zip(active["current_samples"], full):
            active["logits"][key] = value
    return output, loss


def audited_step(forward_step_func, data_iterator, model, optimizer, *args, **kwargs):
    global active
    step = kwargs["iteration"] + 1
    module = unwrap(model[0])
    if os.environ.get("RWKV_DP_CAPTURE_LOADED") == "1":
        torch.save(
            optimizer.state_dict(),
            OUT / f"loaded-optimizer-rank{torch.distributed.get_rank()}.pt",
        )
    active = dict(step=step, samples=[], logits={}, teacher_requests=0)
    torch.cuda.reset_peak_memory_stats()
    real_optimizer_step = optimizer.step

    def capture_update():
        active["grads"] = gradient_snapshot(module, optimizer)
        active["before"] = {
            key: param.detach().cpu().clone()
            for key, param in module.named_parameters()
        }
        result = real_optimizer_step()
        if os.environ.get("RWKV_DP_OPTIMIZER_AUDIT") == "1":
            active["optimizer"] = optimizer_snapshot(optimizer)
        active["after"] = {
            key: param.detach().cpu().clone()
            for key, param in module.named_parameters()
        }
        return result

    optimizer.step = capture_update
    try:
        result = original_step(
            audited_forward, data_iterator, model, optimizer, *args, **kwargs
        )
        active.update(
            memory=dict(
                allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            ),
            tp=ps.get_tensor_model_parallel_rank(),
            pp=ps.get_pipeline_model_parallel_rank(),
            dp=ps.get_data_parallel_rank(),
            cp=ps.get_context_parallel_rank(),
            losses={key: float(value) for key, value in result[0].items()},
            grad_norm=float(result[5]),
            skipped=int(result[1]),
        )
        torch.save(active, OUT / f"step{step}-rank{torch.distributed.get_rank()}.pt")
        return result
    finally:
        optimizer.step = real_optimizer_step
        active = None


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    pretrain.get_batch = audited_batch
    pretrain._get_online_teacher = lambda args: FixtureTeacher()
    training.train_step = audited_step
    with patch(
        "transformers.dynamic_module_utils.resolve_trust_remote_code", return_value=True
    ):
        args = parse_and_validate_args(extra_args_provider=add_rwkv_arguments)
        initialize_runtime_services(args)
        if os.environ.get("RWKV_DP_FP32") == "1":
            original_init = pretrain.RWKV7Model.__init__

            def oracle_init(self, *a, **kw):
                original_init(self, *a, **kw)
                for layer in self.decoder.layers:
                    layer.self_attention.core_attention.forward = recurrence

            pretrain.RWKV7Model.__init__ = oracle_init
        pretrain.pretrain(
            pretrain_cfg_container_from_args(args),
            datasets,
            ModelType.encoder_or_decoder,
            pretrain.forward_step,
            model_provider=pretrain.model_provider,
            callbacks=[StudentPerplexityCallback()],
        )
