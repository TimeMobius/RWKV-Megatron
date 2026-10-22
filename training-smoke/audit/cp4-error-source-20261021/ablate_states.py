"""Isolate CP4 affine summaries with native serial-state boundary interventions."""
import argparse
import importlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from fla.ops.cp import build_cp_context
from fla.ops.rwkv7 import chunk_rwkv7

HERE = Path(__file__).resolve().parent
DPLR = importlib.import_module('fla.ops.generalized_delta_rule.dplr.chunk')
NAMES = ('r', 'w', 'k', 'v', 'a', 'b')
OPTS = dict(scale=1., safe_gate=True, chunk_size=64)


def metric(a, b):
    a, b = a.detach().float(), b.detach().float()
    diff = a-b
    return dict(relative_l2=float(diff.norm()/b.norm().clamp_min(1e-30)),
                max_abs=float(diff.abs().max()), exact=bool(torch.equal(a,b)),
                finite=bool(a.isfinite().all() and b.isfinite().all()))


def native_forward(kw):
    group = kw['context'].group
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    k, v = kw['k'], kw['v']
    carry = torch.zeros((1,k.shape[2],k.shape[3],v.shape[3]),device=k.device,dtype=torch.float32)
    initial = None
    for owner in range(size):
        if rank == owner:
            initial = carry.clone()
            _, _, carry = DPLR.chunk_dplr_fwd_h(
                kg=k, bg=kw['bg'], v=v, w=kw['w'], u=kw['u'], gk=kw['gk'],
                initial_state=carry, output_final_state=True, cu_seqlens=kw['cu_seqlens'],
                chunk_size=kw['chunk_size'])
        dist.broadcast(carry, src=dist.get_process_group_ranks(group)[owner], group=group)
    return initial


def native_backward(kw):
    group = kw['context'].group
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    q, do = kw['q'], kw['do']
    carry = torch.zeros((1,q.shape[2],q.shape[3],do.shape[3]),device=q.device,dtype=torch.float32)
    incoming = None
    for owner in reversed(range(size)):
        if rank == owner:
            incoming = carry.clone()
            _, carry, _ = DPLR.chunk_dplr_bwd_dhu(
                qg=q, bg=kw['bg'], w=kw['w'], gk=kw['gk'], h0=torch.zeros_like(carry),
                dht=carry, do=do, dv=kw['dv'], cu_seqlens=kw['cu_seqlens'], chunk_size=kw['chunk_size'])
        dist.broadcast(carry, src=dist.get_process_group_ranks(group)[owner], group=group)
    return incoming


def load_inputs(layer, tp):
    shards = [torch.load(HERE/f'layer{layer}-rank{2*cp+tp}.pt',weights_only=True) for cp in range(4)]
    full = []
    for name in NAMES:
        x = torch.cat([s['inputs'][name] for s in shards],dim=0)
        x = x.permute(1,0,2).reshape(1,x.shape[0],18,64).cuda().contiguous()
        full.append(-x if name == 'w' else x)
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers',default='0,4,13,23')
    parser.add_argument('--tp',type=int,default=0)
    parser.add_argument('--modes',default='default,native_forward,native_backward,native_both')
    args, _ = parser.parse_known_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    assert dist.get_world_size() == 4
    original_f = DPLR.chunk_gated_delta_rule_fwd_h_pre_process
    original_b = DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process
    records = {}
    for layer in map(int,args.layers.split(',')):
        xs = load_inputs(layer,args.tp)
        length = xs[0].shape[1]
        local_length = length//4
        sl = slice(rank*local_length,(rank+1)*local_length)
        go = torch.sin(torch.arange(xs[0].numel(),device='cuda').float()*0.013).reshape_as(xs[0]).bfloat16()
        full = [x.detach().requires_grad_() for x in xs]
        ref, _ = chunk_rwkv7(*full,**OPTS)
        refs = torch.autograd.grad(ref,full,go)
        shape = (1,xs[0].shape[2],64,64)
        ref_h0 = torch.zeros(shape,device='cuda',dtype=torch.float32)
        if rank:
            _, ref_h0 = chunk_rwkv7(*[x[:,:rank*local_length].contiguous() for x in xs],
                                   output_final_state=True,**OPTS)
        ref_dht = torch.zeros_like(ref_h0)
        if rank < 3:
            end = (rank+1)*local_length
            _, boundary = chunk_rwkv7(*[x[:,:end].contiguous() for x in xs],output_final_state=True,**OPTS)
            boundary = boundary.detach().requires_grad_()
            suffix, _ = chunk_rwkv7(*[x[:,end:].contiguous() for x in xs],initial_state=boundary,**OPTS)
            ref_dht, = torch.autograd.grad(suffix,boundary,go[:,end:].contiguous())
        rows = {}
        for mode in args.modes.split(','):
            boundary_metrics = {}
            def forward_pre(**kw):
                baseline = original_f(**kw)
                native = native_forward(kw)
                boundary_metrics['forward_cp_vs_reference'] = metric(baseline,ref_h0)
                boundary_metrics['forward_native_vs_reference'] = metric(native,ref_h0)
                return native if mode in ('native_forward','native_both') else baseline
            def backward_pre(**kw):
                baseline, _ = original_b(**kw)
                native = native_backward(kw)
                boundary_metrics['backward_cp_vs_reference'] = metric(baseline,ref_dht)
                boundary_metrics['backward_native_vs_reference'] = metric(native,ref_dht)
                return (native if mode in ('native_backward','native_both') else baseline), None
            DPLR.chunk_gated_delta_rule_fwd_h_pre_process = forward_pre
            DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process = backward_pre
            local = [x[:,sl].detach().contiguous().requires_grad_() for x in xs]
            seq = torch.tensor([0,length],device='cuda',dtype=torch.long)
            ctx = build_cp_context(seq,group=dist.group.WORLD,layout='contiguous')
            actual, _ = chunk_rwkv7(*local,cp_context=ctx,**OPTS)
            grads = torch.autograd.grad(actual,local,go[:,sl].contiguous())
            row = dict(output=metric(actual,ref[:,sl]),
                       gradients={name:metric(g,rg[:,sl]) for name,g,rg in zip(NAMES,grads,refs)},
                       boundaries=boundary_metrics)
            rows[mode] = row
            print(f'tp{args.tp} layer{layer} rank{rank} {mode}: output={row["output"]["relative_l2"]:.8g} '
                  f'gradmax={max(g["relative_l2"] for g in row["gradients"].values()):.8g}',flush=True)
        records[str(layer)] = rows
        (HERE/f'ablation-tp{args.tp}-rank{rank}.json').write_text(json.dumps(records,indent=2)+'\n')
        DPLR.chunk_gated_delta_rule_fwd_h_pre_process = original_f
        DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process = original_b
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
