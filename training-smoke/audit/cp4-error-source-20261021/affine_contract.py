"""Measure F(H) != F(0)+M H and isolate state-rounding with local diagnostic kernels.

Only the diagnostic process substitutes ordinary state kernels. Production
files are unmodified. Float32 variants remove recurrent operand BF16 casts and
use IEEE float32 products; they are controls, not a proposed training fix.
"""
import argparse
import importlib
import inspect
import hashlib
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from fla.ops.cp import build_cp_context
from fla.ops.rwkv7 import chunk_rwkv7

from ablate_states import HERE, OPTS, load_inputs

DPLR = importlib.import_module('fla.ops.generalized_delta_rule.dplr.chunk')
CP = importlib.import_module('fla.ops.cp.chunk_delta_h')
FORWARD = importlib.import_module('fla.ops.generalized_delta_rule.dplr.chunk_h_fwd')
BACKWARD = importlib.import_module('fla.ops.generalized_delta_rule.dplr.chunk_h_bwd')
CONTROL_DIR = HERE


def metric(a, b):
    a, b = a.detach().double(), b.detach().double()
    diff = a-b
    return dict(relative_l2=float(diff.norm()/b.norm().clamp_min(1e-30)),
                max_abs=float(diff.abs().max()), exact=bool(torch.equal(a,b)),
                finite=bool(a.isfinite().all() and b.isfinite().all()))


def generated_kernel(module, name, replacements, label='float32_control'):
    kernel = getattr(module,name)
    while not inspect.isfunction(kernel):
        kernel = kernel.fn
    source = inspect.getsource(kernel)
    variant_name = f'{name}_{label}'
    source = source.replace(f'def {name}(',f'def {variant_name}(')
    for old,new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old,new,1)
    path = CONTROL_DIR/f'{variant_name}.py'
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# Diagnostic control derived from installed kernel; not production.\n'+source)
    dist.barrier()
    namespace = dict(vars(module))
    exec(compile(path.read_text(),str(path),'exec'),namespace)
    return namespace[variant_name]


def ideal_matrix(kw, backward=False):
    w = kw['w'][0].permute(1,0,2).double()
    bg = kw['bg'][0].permute(1,0,2).double()
    g = kw['gk'][0].permute(1,0,2).double()
    heads,length,dim = w.shape
    matrix = torch.eye(dim,device=w.device,dtype=torch.float64).expand(heads,dim,dim).clone()
    blocks = range(length//64-1,-1,-1) if backward else range(length//64)
    for block in blocks:
        sl = slice(block*64,(block+1)*64)
        current = torch.diag_embed(torch.exp2(g[:,(block+1)*64-1]))
        current = current + (w[:,sl].transpose(-1,-2) @ bg[:,sl] if backward
                             else bg[:,sl].transpose(-1,-2) @ w[:,sl])
        matrix = current @ matrix
    return matrix


def forward_state(kw, initial, kernel):
    original = FORWARD.chunk_dplr_fwd_kernel_h
    FORWARD.chunk_dplr_fwd_kernel_h = kernel
    try:
        return DPLR.chunk_dplr_fwd_h(kg=kw['k'],bg=kw['bg'],v=kw['v'],w=kw['w'],u=kw['u'],
            gk=kw['gk'],initial_state=initial,output_final_state=True,cu_seqlens=kw['cu_seqlens'],
            chunk_size=kw['chunk_size'])[2]
    finally:
        FORWARD.chunk_dplr_fwd_kernel_h = original


def backward_state(kw, initial, kernel):
    original = BACKWARD.chunk_dplr_bwd_kernel_dhu
    BACKWARD.chunk_dplr_bwd_kernel_dhu = kernel
    try:
        return DPLR.chunk_dplr_bwd_dhu(qg=kw['q'],bg=kw['bg'],w=kw['w'],gk=kw['gk'],
            h0=torch.zeros_like(initial),dht=initial,do=kw['do'],dv=kw['dv'],
            cu_seqlens=kw['cu_seqlens'],chunk_size=kw['chunk_size'])[1]
    finally:
        BACKWARD.chunk_dplr_bwd_kernel_dhu = original


def contract(kw, initial, state_fn, original, control, summary, ieee_bf16=None):
    matrix = ideal_matrix(kw,backward=state_fn is backward_state)
    zero = torch.zeros_like(initial)
    rows = dict(matrix_summary_vs_fp64=metric(summary[:,:,64:].double(),matrix),
                summary_dtype=str(summary.dtype), initial_dtype=str(initial.dtype))
    variants = [('production',original),('float32_control',control)]
    if ieee_bf16 is not None:
        variants.append(('ieee_bf16_state',ieee_bf16))
    for name,kernel in variants:
        base = state_fn(kw,zero,kernel)
        actual = state_fn(kw,initial,kernel)
        half = state_fn(kw,initial*0.5,kernel)
        predicted = base.double() + (matrix @ initial.double()).reshape_as(base)
        cp_predicted = summary[:,:,:64].unsqueeze(0).double() + summary[:,:,64:].double() @ initial.double()
        rows[name] = dict(
            affine_residual=metric(actual.double(),predicted),
            cp_affine_residual=metric(actual.double(),cp_predicted),
            matrix_precision_error_relative_to_state=float((cp_predicted-predicted).norm()/actual.double().norm().clamp_min(1e-30)),
            midpoint_residual=metric(actual.double(),2*half.double()-base.double()),
            zero_state_vs_summary=metric(base,summary[:,:,:64].unsqueeze(0)))
    return rows


def main():
    global CONTROL_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp',type=int,default=0)
    parser.add_argument('--layers',default='4,13')
    parser.add_argument('--affine-precision',choices=('default','tf32x3','ieee'),default='default')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    CONTROL_DIR = HERE/f'generated-{args.affine_precision}-tp{args.tp}'
    if args.affine_precision == 'ieee':
        class IEEEMatrixChain:
            def __init__(self,kernel):
                self.kernel = kernel
            def __getitem__(self,grid):
                def launch(*a,**kw):
                    kw['AFFINE_CHAIN_PRECISION'] = 'ieee'
                    return self.kernel[grid](*a,**kw)
                return launch
        for name in ('pre_process_fwd_kernel_merged','pre_process_bwd_kernel_merged','merge_fwd_bwd_kernel'):
            setattr(CP,name,IEEEMatrixChain(getattr(CP,name)))
    f_orig = FORWARD.chunk_dplr_fwd_kernel_h
    b_orig = BACKWARD.chunk_dplr_bwd_kernel_dhu
    f_control = generated_kernel(FORWARD,'chunk_dplr_fwd_kernel_h',[
        ('tl.dot(b_w, b_h.to(b_w.dtype))',
         'tl.dot(b_w.to(tl.float32), b_h, input_precision="ieee")'),
        ('tl.dot(b_bg.to(b_hc.dtype), b_v2, b_hc)',
         'tl.dot(b_bg.to(tl.float32), b_v2, b_hc, input_precision="ieee")')])
    f_ieee_bf16 = generated_kernel(FORWARD,'chunk_dplr_fwd_kernel_h',[
        ('tl.dot(b_bg.to(b_hc.dtype), b_v2, b_hc)',
         'tl.dot(b_bg.to(tl.float32), b_v2, b_hc, input_precision="ieee")')],label='ieee_bf16_state')
    b_control = generated_kernel(BACKWARD,'chunk_dplr_bwd_kernel_dhu',[
        ('b_dv + tl.dot(b_bg, b_dh.to(b_bg.dtype))',
         'b_dv.to(tl.float32) + tl.dot(b_bg.to(tl.float32), b_dh, input_precision="ieee")'),
        ('tl.dot(b_w, b_dv2.to(b_qg.dtype), b_dh_tmp)',
         'tl.dot(b_w.to(tl.float32), b_dv2, b_dh_tmp, input_precision="ieee")')])
    pre_f, pre_b = DPLR.chunk_gated_delta_rule_fwd_h_pre_process,DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process
    gather = CP.all_gather_into_tensor
    current, records = {}, {}
    def capture_gather(*a,**kw):
        result = gather(*a,**kw)
        current['summaries'] = result[0].detach().clone()
        # Every received rank summary must be bitwise equal to its sender.
        local = a[0] if a else kw['input']
        assert torch.equal(result[0][rank],local)
        def fingerprint(tensor):
            return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()
        hashes = [None]*dist.get_world_size()
        dist.all_gather_object(hashes,fingerprint(local))
        assert all(fingerprint(result[0][i]) == expected for i,expected in enumerate(hashes))
        return result
    CP.all_gather_into_tensor = capture_gather
    for layer in map(int,args.layers.split(',')):
        xs = load_inputs(layer,args.tp)
        length = xs[0].shape[1]
        local_length = length//4
        sl = slice(rank*local_length,(rank+1)*local_length)
        go = torch.sin(torch.arange(xs[0].numel(),device='cuda').float()*0.013).reshape_as(xs[0]).bfloat16()
        h0 = torch.zeros((1,18,64,64),device='cuda',dtype=torch.float32)
        if rank:
            _,h0 = chunk_rwkv7(*[x[:,:rank*local_length].contiguous() for x in xs],output_final_state=True,**OPTS)
        dht = torch.zeros_like(h0)
        if rank < 3:
            end = (rank+1)*local_length
            _,boundary = chunk_rwkv7(*[x[:,:end].contiguous() for x in xs],output_final_state=True,**OPTS)
            boundary = boundary.detach().requires_grad_()
            suffix,_ = chunk_rwkv7(*[x[:,end:].contiguous() for x in xs],initial_state=boundary,**OPTS)
            dht, = torch.autograd.grad(suffix,boundary,go[:,end:].contiguous())
        result = {}
        def forward_pre(**kw):
            output = pre_f(**kw)
            summary = current['summaries'][rank]
            if rank < 3:
                result['forward'] = contract(kw,h0,forward_state,f_orig,f_control,summary,f_ieee_bf16)
            return output
        def backward_pre(**kw):
            output = pre_b(**kw)
            summary = current['summaries'][rank]
            if rank > 0:
                result['backward'] = contract(kw,dht,backward_state,b_orig,b_control,summary)
            return output
        DPLR.chunk_gated_delta_rule_fwd_h_pre_process = forward_pre
        DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process = backward_pre
        seq = torch.tensor([0,length],device='cuda',dtype=torch.long)
        ctx = build_cp_context(seq,group=dist.group.WORLD,layout='contiguous',
                               use_tf32x3_affine_chain=args.affine_precision == 'tf32x3')
        local = [x[:,sl].detach().contiguous().requires_grad_() for x in xs]
        output,_ = chunk_rwkv7(*local,cp_context=ctx,**OPTS)
        torch.autograd.grad(output,local,go[:,sl].contiguous())
        result['all_gather_hashes_exact'] = True
        records[str(layer)] = result
        (HERE/f'contract-{args.affine_precision}-tp{args.tp}-rank{rank}.json').write_text(json.dumps(records,indent=2)+'\n')
        print(f'tp{args.tp} layer{layer} rank{rank} {args.affine_precision}: completed',flush=True)
        DPLR.chunk_gated_delta_rule_fwd_h_pre_process,DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process = pre_f,pre_b
    CP.all_gather_into_tensor = gather
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
