"""Isolate CP WKV errors using identical real inputs and output gradients.

Two-rank test; CP1 reference is evaluated independently on both ranks.
The native_state variant substitutes the ordinary prefix kernel's FP32 final
state for the CP affine-summary initial state (forward diagnostic only).
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from fla.ops.cp import build_cp_context
from fla.ops.rwkv7 import chunk_rwkv7

ROOT=Path(__file__).resolve().parent
DPLR=importlib.import_module('fla.ops.generalized_delta_rule.dplr.chunk')


def metric(x,y):
    x,y=x.detach().double(),y.detach().double()
    return {'max_abs':(x-y).abs().max().item(),'rel_l2':((x-y).norm()/y.norm().clamp_min(1e-30)).item(), 'nonzero':int(torch.count_nonzero(x-y)), 'elements':x.numel()}


def recurrence(xs):
    r,g,k,v,a,b=[x.double() for x in xs]
    h=torch.zeros((r.shape[0],r.shape[2],r.shape[3],v.shape[3]),dtype=torch.float64,device=r.device)
    outputs=[]
    midpoint=None
    for t in range(r.shape[1]):
        correction=(a[:,t,...,None]*h).sum(-2)
        h=g[:,t,...,None].exp()*h + b[:,t,...,None]*correction[...,None,:] + k[:,t,...,None]*v[:,t,...,None,:]
        outputs.append((r[:,t,...,None]*h).sum(-2))
        if t+1==r.shape[1]//2:
            midpoint=h.clone()
    return torch.stack(outputs,1),midpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layers',default='0,6,23')
    parser.add_argument('--dtypes',default='bf16')
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl')
    records={}
    source={pp:torch.load(ROOT/f'cp1/trace-tp0-pp{pp}-cp0.pt',weights_only=False) for pp in (0,1)}
    original_pre=DPLR.chunk_gated_delta_rule_fwd_h_pre_process
    for layer in map(int,args.layers.split(',')):
        tensors=source[int(layer>=12)]
        for dtype_name in args.dtypes.split(','):
            dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[dtype_name]
            xs=[]
            for name in ('r','w','k','v','a','b'):
                x=tensors[f'{layer}.input_{name}'].permute(1,0,2).reshape(1,-1,2,64).to(device='cuda',dtype=dtype).contiguous()
                xs.append(-x if name=='w' else x)
            length=xs[0].shape[1]
            mid=length//2
            sl=slice(rank*mid,(rank+1)*mid)
            torch.manual_seed(20261021)
            # Same BF16-exact incoming derivative for all dtype variants.
            go=torch.randn(xs[0].shape,device='cuda',dtype=torch.bfloat16).to(dtype)
            opts=dict(scale=1.,safe_gate=True,chunk_size=64)
            full=[x.clone().requires_grad_() for x in xs]
            ref,_=chunk_rwkv7(*full,**opts)
            ref_grad=torch.autograd.grad(ref,full,go)
            prefix,prefix_state=chunk_rwkv7(*[x[:,:mid].contiguous() for x in xs],output_final_state=True,**opts)
            seq=torch.tensor([0,length],device='cuda',dtype=torch.long)
            varlen,_=chunk_rwkv7(*xs,cu_seqlens=seq,**opts)
            suffix,_=chunk_rwkv7(*[x[:,mid:].contiguous() for x in xs],initial_state=prefix_state,**opts)
            result={'varlen_output':metric(varlen,ref),'native_split_output':metric(torch.cat((prefix,suffix),1),ref)}
            # Serial ordinary kernels connected through differentiable FP32 state.
            split_inputs=[x.clone().requires_grad_() for x in xs]
            p,ht=chunk_rwkv7(*[x[:,:mid].contiguous() for x in split_inputs],output_final_state=True,**opts)
            s,_=chunk_rwkv7(*[x[:,mid:].contiguous() for x in split_inputs],initial_state=ht,**opts)
            split_grads=torch.autograd.grad(torch.cat((p,s),1),split_inputs,go)
            result['native_split_grads']={name:metric(a,b) for name,a,b in zip(('r','w','k','v','a','b'),split_grads,ref_grad)}
            if layer==0 and dtype_name=='bf16':
                high,high_state=recurrence(xs)
                result['fp64_reference']={'cp1_output':metric(ref,high),'cp1_prefix_state':metric(prefix_state,high_state)}
            for mode in ('default','tf32x3','native_state'):
                ctx=build_cp_context(seq,group=dist.group.WORLD,layout='contiguous',use_tf32x3_affine_chain=mode=='tf32x3')
                state_metrics={}
                def capture_pre(**kw):
                    h=original_pre(**kw)
                    expected=prefix_state if rank else torch.zeros_like(h)
                    state_metrics.update(metric(h,expected))
                    if mode=='native_state':
                        return expected
                    return h
                DPLR.chunk_gated_delta_rule_fwd_h_pre_process=capture_pre
                local=[x[:,sl].contiguous().detach().requires_grad_() for x in xs]
                output,_=chunk_rwkv7(*local,cp_context=ctx,**opts)
                entry={'output':metric(output,ref[:,sl]),'state':state_metrics.copy(),
                       'first64':metric(output[:,:64],ref[:,sl][:,:64]),
                       'after256':metric(output[:,256:],ref[:,sl][:,256:])}
                if mode!='native_state':
                    grads=torch.autograd.grad(output,local,go[:,sl].contiguous())
                    entry['grads']={name:metric(a,b[:,sl]) for name,a,b in zip(('r','w','k','v','a','b'),grads,ref_grad)}
                result[mode]=entry
                DPLR.chunk_gated_delta_rule_fwd_h_pre_process=original_pre
            records[f'layer{layer}.{dtype_name}']=result
            (ROOT/f'replay-rank{rank}.json').write_text(json.dumps(records,indent=2)+'\n')
            print(f'layer {layer} {dtype_name} rank{rank}: output {result["default"]["output"]}, native {result["native_state"]["output"]}',flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__=='__main__':main()
