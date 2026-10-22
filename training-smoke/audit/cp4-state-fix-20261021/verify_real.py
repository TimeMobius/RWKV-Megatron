"""Strict replay through the production recurrent CP route, no kernel substitution."""
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from fla.ops.cp import build_cp_context
from fla.ops.rwkv7 import chunk_rwkv7

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('state_diagnostic',HERE.parent/'cp4-error-source-20261021/ablate_states.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group('nccl')
    records = []
    for tp in (0,1):
        for layer in (0,4,13,23):
            xs = data.load_inputs(layer,tp)
            go = torch.sin(torch.arange(xs[0].numel(),device='cuda').float()*0.013).reshape_as(xs[0]).bfloat16()
            full = [x.detach().requires_grad_() for x in xs]
            ref,_ = chunk_rwkv7(*full,**data.OPTS)
            refg = torch.autograd.grad(ref,full,go)
            size = xs[0].shape[1]//4
            sl = slice(rank*size,(rank+1)*size)
            local = [x[:,sl].detach().contiguous().requires_grad_() for x in xs]
            seq = torch.tensor([0,xs[0].shape[1]],device='cuda',dtype=torch.long)
            context = replace(build_cp_context(seq,group=dist.group.WORLD,layout='contiguous'),dplr_state_mode='recurrent')
            actual,_ = chunk_rwkv7(*local,cp_context=context,**data.OPTS)
            grads = torch.autograd.grad(actual,local,go[:,sl].contiguous())
            row = dict(tp=tp,layer=layer,output=data.metric(actual,ref[:,sl]),
                       gradients={n:data.metric(g,r[:,sl]) for n,g,r in zip(data.NAMES,grads,refg)})
            assert row['output']['exact'] and all(g['exact'] for g in row['gradients'].values())
            records.append(row)
            print(f'CP{rank} TP{tp} layer{layer}: production output/all gradients exact',flush=True)
    (HERE/f'real-operator-rank{rank}.json').write_text(json.dumps(records,indent=2)+'\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
