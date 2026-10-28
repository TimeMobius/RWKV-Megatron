"""Record per-rank memory and actual Adam updates without copying a large model."""

import json
import os
from pathlib import Path
import runpy
import time
from collections import defaultdict

import torch

import megatron.training.training as training

OUT = Path(os.environ["RWKV_CAPACITY_OUTPUT"])
OUT.mkdir(parents=True, exist_ok=True)
RANK = int(os.environ["RANK"])
BUILD = training.get_megatron_optimizer
TRAIN_STEP = training.train_step


def emit(event, **values):
    torch.cuda.synchronize()
    row = dict(event=event, rank=RANK, allocated_mib=torch.cuda.memory_allocated() / 2**20,
               reserved_mib=torch.cuda.memory_reserved() / 2**20,
               peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, **values)
    with (OUT / f"rank{RANK}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def parts(optimizer):
    for child in getattr(optimizer, "chained_optimizers", [optimizer]):
        yield child


def storage(optimizer):
    counts, seen = defaultdict(int), set()

    def record(value, name):
        if not isinstance(value, torch.Tensor):
            return
        key = (str(value.device), value.untyped_storage().data_ptr())
        if key not in seen:
            seen.add(key)
            counts[f"{name}/{value.device.type}/{value.dtype}"] += value.untyped_storage().nbytes()

    for part in parts(optimizer):
        for buffer in getattr(part, "buffers", []):
            record(buffer.param_data, "model_buffer")
            record(buffer.grad_data, "gradient_buffer")
        inner = part.optimizer
        for sub in getattr(inner, "sub_optimizers", [inner]):
            for group in sub.param_groups:
                for param in group["params"]:
                    record(param, "master_parameter")
            for state in sub.state.values():
                for key, value in state.items():
                    record(value, key)
    return dict(counts)


def build_optimizer(*args, **kwargs):
    optimizer = BUILD(*args, **kwargs)
    emit("optimizer_built", storage_bytes=storage(optimizer))
    original_step, updates = optimizer.step, 0

    def step(*a, **kw):
        nonlocal updates
        updates += 1
        snapshots = []
        if updates <= 2:
            for part in parts(optimizer):
                inner = part.optimizer
                for sub in getattr(inner, "sub_optimizers", [inner]):
                    parameters = sorted((p for g in sub.param_groups for p in g["params"]),
                                        key=lambda p: p.numel(), reverse=True)[:3]
                    for param in parameters:
                        count = min(1024, param.numel())
                        # Float32 linspace rounds a multi-billion-element
                        # endpoint up to numel, producing an invalid CUDA index.
                        indices = torch.arange(count, device=param.device, dtype=torch.long)
                        indices = indices * (param.numel() - 1) // max(count - 1, 1)
                        snapshots.append((param, indices, param.detach().flatten()[indices].clone()))
        result = original_step(*a, **kw)
        emit("optimizer_update", update=updates, storage_bytes=storage(optimizer),
             sampled_master_updates=[dict(device=str(p.device), dtype=str(p.dtype),
                                          max_abs_delta=(p.detach().flatten()[indices] - old).abs().max().item())
                                     for p, indices, old in snapshots])
        return result

    optimizer.step = step
    return optimizer


def train_step(*args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        result = TRAIN_STEP(*args, **kwargs)
        emit("train_step", iteration=kwargs["iteration"] + 1,
             seconds=time.monotonic() - started, skipped=int(result[1]),
             grad_norm=float(result[5]))
        return result
    except torch.OutOfMemoryError:
        emit("oom", iteration=kwargs["iteration"] + 1)
        raise


if __name__ == "__main__":
    training.get_megatron_optimizer = build_optimizer
    training.train_step = train_step
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "pretrain_rwkv7.py"), run_name="__main__")
