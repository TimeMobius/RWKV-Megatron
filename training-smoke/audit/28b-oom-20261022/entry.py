"""Read-only memory/update instrumentation for the real 28B training path."""
import json
import os
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '3rdparty/Megatron-LM-0.11.0'))
import megatron.training.training as training

AUDIT = Path(os.environ['RWKV_MEMORY_AUDIT'])
AUDIT.mkdir(parents=True, exist_ok=True)
rank = int(os.environ['RANK'])


def emit(event, **payload):
    torch.cuda.synchronize()
    row = dict(event=event, rank=rank,
               allocated=torch.cuda.memory_allocated(),
               reserved=torch.cuda.memory_reserved(),
               max_allocated=torch.cuda.max_memory_allocated(), **payload)
    with (AUDIT / f'rank{rank}.jsonl').open('a') as out:
        out.write(json.dumps(row) + '\n')


def parts(opt):
    children = getattr(opt, 'chained_optimizers', None)
    if children is None:
        yield opt
    else:
        for child in children:
            yield from parts(child)


def measure(opt):
    counts = defaultdict(int)
    seen = set()

    def record(tensor, label):
        if not isinstance(tensor, torch.Tensor):
            return
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if key in seen:
            return
        seen.add(key)
        counts[f'{label}/{tensor.device.type}/{tensor.dtype}'] += tensor.untyped_storage().nbytes()

    for part in parts(opt):
        for buf in getattr(part, 'buffers', []):
            record(buf.param_data, 'model_buffer')
            record(buf.grad_data, 'gradient_buffer')
        inner = getattr(part, 'optimizer', None)
        if inner is None:
            continue
        for group in inner.param_groups:
            for p in group['params']:
                record(p, 'optimizer_parameter')
        for sub in getattr(inner, 'sub_optimizers', []):
            for group in sub.param_groups:
                for param in group['params']:
                    record(param, 'master_param')
        for state in inner.state.values():
            for name, tensor in state.items():
                record(tensor, name)
    return dict(counts)


original_optimizer = training.get_megatron_optimizer


def build_optimizer(*args, **kwargs):
    opt = original_optimizer(*args, **kwargs)
    emit('optimizer_built', storage_bytes=measure(opt),
         sharding_group_sizes=[p.data_parallel_group.size() for p in parts(opt)
                               if hasattr(p, 'data_parallel_group')])
    original_step = opt.step
    steps = 0

    def step(*args, **kwargs):
        nonlocal steps
        steps += 1
        if steps <= 2:
            emit('before_step', step=steps, storage_bytes=measure(opt))
        samples = []
        if steps <= 2:
            for part in parts(opt):
                inner = getattr(part, 'optimizer', None)
                if inner is None:
                    continue
                optimizers = getattr(inner, 'sub_optimizers', [inner])
                for sub in optimizers:
                    candidates = [p for group in sub.param_groups for p in group['params']]
                    if candidates:
                        p = max(candidates, key=lambda p: p.numel())
                        samples.append((p, p.detach().view(-1)[:4096].clone()))
        try:
            result = original_step(*args, **kwargs)
        except torch.OutOfMemoryError:
            emit('oom', step=steps, storage_bytes=measure(opt))
            raise
        if steps <= 2 or steps % 10 == 0:
            emit('after_step', step=steps, storage_bytes=measure(opt),
                 master_update_samples=[dict(device=str(p.device), dtype=str(p.dtype),
                     max_abs_delta=(p.detach().view(-1)[:4096] - old).abs().max().item())
                     for p, old in samples])
        return result

    opt.step = step
    return opt


training.get_megatron_optimizer = build_optimizer
runpy.run_path(str(ROOT / 'pretrain_rwkv7.py'), run_name='__main__')
