"""Capture complete real WKV inputs at the CP4 worst-error layers, first microbatch."""
import os
import runpy
from pathlib import Path

import torch
import torch.distributed as dist
import megatron.training.training as training
from megatron.core import parallel_state as ps
from megatron.core.utils import unwrap_model

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ['CP4_DIAG_OUT'])
ORIGINAL = training.train_step


def train_step(*args, **kwargs):
    handles, captured = [], set()
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args[2]:
        for layer in unwrap_model(model).decoder.layers:
            number = layer.layer_number - 1
            if number not in (0, 4, 13, 23):
                continue
            def hook(module, inputs, output, number=number):
                if number not in captured:
                    captured.add(number)
                    torch.save(dict(layer=number, tp=ps.get_tensor_model_parallel_rank(),
                                    cp=ps.get_context_parallel_rank(),
                                    inputs={name:x.detach().cpu().clone() for name,x in
                                            zip(('r','k','v','w','a','b'),inputs)}),
                               OUT/f'layer{number}-rank{dist.get_rank()}.pt')
            handles.append(layer.self_attention.core_attention.register_forward_hook(hook))
    optimizer = args[3]
    original_step = optimizer.step
    optimizer.step = lambda *a, **kw: (True,0.,None)
    try:
        return ORIGINAL(*args, **kwargs)
    finally:
        optimizer.step = original_step
        for handle in handles:
            handle.remove()


if __name__ == '__main__':
    training.train_step = train_step
    runpy.run_path(str(ROOT/'pretrain_rwkv7.py'),run_name='__main__')
