"""Four-rank controlled default/tf32x3 affine-chain replay of real WKV inputs."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from fla.ops.cp import build_cp_context
from fla.ops.rwkv7 import chunk_rwkv7

HERE = Path(__file__).resolve().parent
NAMES = ('r', 'w', 'k', 'v', 'a', 'b')


def metric(a, b):
    a, b = a.detach().float(), b.detach().float()
    return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                max_abs=float((a-b).abs().max()), exact=torch.equal(a,b))


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(8)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 4
    records = {}
    for layer in (0, 6, 12, 23):
        source = torch.load(HERE.parent/f'cp-error-source-20261021/cp1/trace-tp0-pp{int(layer>=12)}-cp0.pt',
                            weights_only=True, mmap=True)
        xs = []
        for name in NAMES:
            x = source[f'{layer}.input_{name}'].permute(1,0,2).reshape(1,-1,2,64).cuda().contiguous()
            xs.append((-x if name == 'w' else x).requires_grad_())
        length = xs[0].shape[1]
        local_length = length//4
        sl = slice(rank*local_length,(rank+1)*local_length)
        go = torch.sin(torch.arange(xs[0].numel(),device='cuda').float()*0.013).reshape_as(xs[0]).bfloat16()
        ref, _ = chunk_rwkv7(*xs, scale=1., safe_gate=True, chunk_size=64)
        refg = torch.autograd.grad(ref, xs, go)
        seq = torch.tensor([0,length], device='cuda', dtype=torch.long)
        rows = {}
        for mode in (False, True):
            ctx = build_cp_context(seq, group=dist.group.WORLD, layout='contiguous', use_tf32x3_affine_chain=mode)
            local = [x[:,sl].detach().contiguous().requires_grad_() for x in xs]
            actual, _ = chunk_rwkv7(*local, scale=1., safe_gate=True, chunk_size=64, cp_context=ctx)
            grads = torch.autograd.grad(actual, local, go[:,sl].contiguous())
            rows['tf32x3' if mode else 'default'] = dict(output=metric(actual,ref[:,sl]),
                gradients={name: metric(g,rg[:,sl]) for name,g,rg in zip(NAMES,grads,refg)})
        records[str(layer)] = rows
        (HERE/f'affine-rank{rank}.json').write_text(json.dumps(records,indent=2)+'\n')
        print(rank, layer, rows, flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
