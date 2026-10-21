"""Diagnostic-only CP2 boundary interventions. No production source edits."""
import importlib
import inspect
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from fla.ops.cp import build_cp_context
from fla.ops.rwkv7 import chunk_rwkv7
from replay_wkv import metric

ROOT=Path(__file__).resolve().parent
DPLR=importlib.import_module('fla.ops.generalized_delta_rule.dplr.chunk')
CP=importlib.import_module('fla.ops.cp.chunk_delta_h')


def native_forward(**kw):
    assert dist.get_world_size(kw['context'].group)==2
    _,_,ht=DPLR.chunk_dplr_fwd_h(kg=kw['k'],bg=kw['bg'],v=kw['v'],w=kw['w'],u=kw['u'],gk=kw['gk'],initial_state=None,output_final_state=True,cu_seqlens=kw['cu_seqlens'],chunk_size=kw['chunk_size'])
    states=[torch.empty_like(ht) for _ in range(2)]
    dist.all_gather(states,ht,group=kw['context'].group)
    return states[0] if dist.get_rank(kw['context'].group) else torch.zeros_like(ht)


def native_backward(**kw):
    _,dh0,_=DPLR.chunk_dplr_bwd_dhu(qg=kw['q'],bg=kw['bg'],w=kw['w'],gk=kw['gk'],h0=kw['initial_state'],dht=None,do=kw['do'],dv=kw['dv'],cu_seqlens=kw['cu_seqlens'],chunk_size=kw['chunk_size'])
    states=[torch.empty_like(dh0) for _ in range(2)]
    dist.all_gather(states,dh0,group=kw['context'].group)
    # CP has no differentiable external h0; the saved state was used only for
    # forward recomputation. Match the production contract and suppress dh0.
    return (states[1] if dist.get_rank(kw['context'].group)==0 else torch.zeros_like(dh0)),None


def fp32_v2_variant():
    # Derive a reviewable test kernel from the installed source. Only the
    # DPLR he branch changes; only K=64 is exercised by this diagnostic.
    kernel=CP.pre_process_fwd_kernel_merged
    while not inspect.isfunction(kernel):
        kernel=kernel.fn
    source=inspect.getsource(kernel)
    source=source.replace('def pre_process_fwd_kernel_merged(', 'def pre_process_fwd_kernel_merged_fp32_v2(')
    assert '            b_v = b_v.to(k.dtype.element_ty)' in source
    source=source.replace('            b_v = b_v.to(k.dtype.element_ty)', '            if not USE_BG:\n                b_v = b_v.to(k.dtype.element_ty)')
    assert 'tl.dot(b_bg, b_v)' in source
    source=source.replace('tl.dot(b_bg, b_v)', 'tl.dot(b_bg.to(tl.float32), b_v)')
    path=ROOT/'kernel_fp32_v2.py'
    if dist.get_rank()==0:
        path.write_text('# Generated diagnostic variant; do not import as production code.\n'+source)
    dist.barrier()
    ns=dict(vars(CP))
    exec(compile(path.read_text(),str(path),'exec'),ns)
    return ns['pre_process_fwd_kernel_merged_fp32_v2']


def main():
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl')
    original_f=DPLR.chunk_gated_delta_rule_fwd_h_pre_process
    original_b=DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process
    original_kernel=CP.pre_process_fwd_kernel_merged
    variant=fp32_v2_variant()
    records={}
    for layer in (0,6,23):
        trace=torch.load(ROOT/f'cp1/trace-tp0-pp{int(layer>=12)}-cp0.pt',weights_only=False)
        xs=[trace[f'{layer}.input_{n}'].permute(1,0,2).reshape(1,-1,2,64).cuda().contiguous() for n in ('r','w','k','v','a','b')]
        xs[1]=-xs[1]
        mid=xs[0].shape[1]//2
        sl=slice(rank*mid,(rank+1)*mid)
        seq=torch.tensor([0,2*mid],device='cuda',dtype=torch.long)
        opts=dict(scale=1.,safe_gate=True,chunk_size=64)
        torch.manual_seed(20261021)
        go=torch.randn_like(xs[0])
        full=[x.clone().requires_grad_() for x in xs]
        ref,_=chunk_rwkv7(*full,**opts)
        refs=torch.autograd.grad(ref,full,go)
        result={}
        for mode in ('default','fp32_v2','native_forward','native_both'):
            DPLR.chunk_gated_delta_rule_fwd_h_pre_process=native_forward if mode.startswith('native') else original_f
            DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process=native_backward if mode=='native_both' else original_b
            CP.pre_process_fwd_kernel_merged=variant if mode=='fp32_v2' else original_kernel
            ctx=build_cp_context(seq,group=dist.group.WORLD,layout='contiguous')
            local=[x[:,sl].detach().contiguous().requires_grad_() for x in xs]
            out,_=chunk_rwkv7(*local,cp_context=ctx,**opts)
            grads=torch.autograd.grad(out,local,go[:,sl].contiguous())
            result[mode]={'output':metric(out,ref[:,sl]),'grads':{n:metric(a,b[:,sl]) for n,a,b in zip(('r','w','k','v','a','b'),grads,refs)}}
            print(f'layer{layer} rank{rank} {mode}: {result[mode]}',flush=True)
        records[str(layer)]=result
        (ROOT/f'ablation-rank{rank}.json').write_text(json.dumps(records,indent=2)+'\n')
    DPLR.chunk_gated_delta_rule_fwd_h_pre_process=original_f
    DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process=original_b
    CP.pre_process_fwd_kernel_merged=original_kernel
    dist.destroy_process_group()

if __name__=='__main__': main()
