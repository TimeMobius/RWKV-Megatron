"""Evaluate unused FLA operators on real current 3B tensors, audit only."""
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import einsum

from check import metric
from rwkv_megatron.rwkv7_layer_specs import mix_inputs
from fla.modules.layernorm import group_norm
from fla.ops.rwkv7.gate_output_correction import gate_output_correction
from fla.ops.rwkv7 import chunk_rwkv7
from distributed import recurrence

PATH=Path(__file__).resolve().parent
MODEL='/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth'


def correction_current(o,r,k,v,rho,g):
    shaped=lambda t:t.reshape(*t.shape[:-1],-1,64)
    return (o+((shaped(r)*shaped(k)*rho.reshape(-1,64)).sum(-1,keepdim=True)*shaped(v)).flatten(2))*g


def correction_fla(o,r,k,v,rho,g):
    shaped=lambda t:t.reshape(*t.shape[:-1],-1,64)
    return gate_output_correction(o,shaped(r),shaped(k),rho.reshape(-1,64),shaped(v),g)


def norm_current(x,weight,bias):
    return F.group_norm(x.reshape(-1,x.shape[-1]),x.shape[-1]//64,weight,bias,64e-5).reshape_as(x)


def norm_fla(x,weight,bias):
    return group_norm(x,weight,bias,eps=64e-5,num_groups=x.shape[-1]//64)


def norm_formula(x,weight,bias):
    heads=x.reshape(*x.shape[:-1],-1,64)
    centered=heads-heads.mean(-1,keepdim=True)
    norm=centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+64e-5)
    return norm.flatten(2)*weight+bias


def accuracy(inputs,funcs,formula,seed):
    torch.manual_seed(seed)
    oracle=[t.detach().double().requires_grad_() for t in inputs]
    expected=formula(*oracle)
    dy=torch.randn_like(expected).bfloat16()
    expected_grad=torch.autograd.grad(expected,oracle,dy.double())
    rows={}
    for name,fn in funcs.items():
        tensors=[t.detach().requires_grad_() for t in inputs]
        out=fn(*tensors);grads=torch.autograd.grad(out,tensors,dy)
        output=metric(out,expected)
        gradients=[metric(a,b) for a,b in zip(grads,expected_grad)]
        assert output['finite'] and all(m['finite'] for m in gradients)
        rows[name]={'output':output,'gradient':gradients}
    return rows


def bench(inputs,funcs,sequence_lengths=(128,512,2048)):
    rows={}
    for seq in sequence_lengths:
        tensors=[(t.repeat(1,seq//128,*([1]*(t.ndim-2))) if t.ndim>=3 else t).detach().requires_grad_() for t in inputs]
        dy=torch.randn_like(tensors[0]);samples={n:[] for n in funcs};saved_bytes={};kernels={}
        for n,fn in funcs.items():
            for _ in range(4):torch.autograd.grad(fn(*tensors),tensors,dy)
        for round_index in range(5):
            names=list(funcs);names=names[round_index%len(names):]+names[:round_index%len(names)]
            for n in names:
                fn=funcs[n];torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();start.record()
                for _ in range(20):torch.autograd.grad(fn(*tensors),tensors,dy)
                end.record();torch.cuda.synchronize()
                samples[n].append({'gpu_ms':start.elapsed_time(end)/20,'wall_ms':1000*(time.perf_counter()-wall)/20})
        for n,fn in funcs.items():
            input_storages={t.untyped_storage().data_ptr() for t in tensors};saved={}
            def pack(t):
                s=t.untyped_storage()
                if s.data_ptr() not in input_storages:saved[s.data_ptr()]=s.nbytes()
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):out=fn(*tensors)
            torch.autograd.grad(out,tensors,dy);saved_bytes[n]=sum(saved.values())
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
                out=fn(*tensors)
            kernels[n]=[e.name for e in profiler.events() if e.device_type==torch.autograd.DeviceType.CUDA]
        rows[str(seq)]={n:{'samples':samples[n],'median_wall_ms':statistics.median(s['wall_ms'] for s in samples[n]),
                          'median_gpu_ms':statistics.median(s['gpu_ms'] for s in samples[n]),
                          'extra_saved_bytes':saved_bytes[n],'forward_cuda_kernel_count':len(kernels[n]),
                          'forward_cuda_kernels':kernels[n]} for n in funcs}
        print('CANDIDATE BENCH',seq,{n:r['median_wall_ms'] for n,r in rows[str(seq)].items()},flush=True)
    return rows


def wkv_probe(trace,layer):
    inputs=[trace[f'{layer}.core-{n}'].cuda().bfloat16().reshape(1,128,40,64) for n in ('r','k','v','decay','a','b')]
    # Production stores positive raw decay, wrapper negates it.
    inputs[3]=-inputs[3]
    def run(*values,disable=False):
        r,k,v,w,a,b=values
        return chunk_rwkv7(r,w,k,v,a,b,scale=1.,safe_gate=True,chunk_size=64,
                          lower_bound=-0.6065306597126334,disable_recompute=disable)[0]
    funcs={'recompute':lambda *t:run(*t), 'save_intermediates':lambda *t:run(*t,disable=True)}
    baseline=None;accuracy_rows={}
    torch.manual_seed(1900+layer);dy=torch.randn_like(inputs[0])
    oracle=[t.detach().float().requires_grad_() for t in inputs]
    # independent recurrence uses [T,B,D], positive decay; FP32 state.
    packed=[t.flatten(2).transpose(0,1) for t in oracle]
    packed[3]=-packed[3]
    reference=recurrence(*packed).transpose(0,1).reshape_as(inputs[0])
    refgrad=torch.autograd.grad(reference,oracle,dy.float())
    for name,fn in funcs.items():
        tensors=[t.detach().requires_grad_() for t in inputs]
        out=fn(*tensors);grad=torch.autograd.grad(out,tensors,dy)
        row={'output':metric(out,reference),'gradient':[metric(a,b) for a,b in zip(grad,refgrad)]}
        assert row['output']['finite'] and row['output']['relative_l2']<.007
        assert all(m['finite'] and m['relative_l2']<.008 for m in row['gradient'])
        if baseline is None:baseline=(out.detach(),[t.detach() for t in grad])
        else:
            row['vs_recompute']={'forward_exact':torch.equal(out,baseline[0]),
                'gradient_exact':all(torch.equal(a,b) for a,b in zip(grad,baseline[1])),
                'gradient':[metric(a,b) for a,b in zip(grad,baseline[1])]}
        accuracy_rows[name]=row
    return {'accuracy':accuracy_rows,'benchmark':bench(inputs,funcs)}


def main():
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(MODEL,map_location='cpu',weights_only=True,mmap=True)
    cap=torch.load(PATH/'full-production-key-fusion.pth',map_location='cpu',weights_only=True,mmap=True)
    trace=cap['trace'];accuracy_rows={};bench_inputs={}
    funcs={'correction':{'current':correction_current,'fla':correction_fla},
           'group_norm':{'current':norm_current,'fla':norm_fla}}
    for layer in (0,15,31):
        prefix=f'blocks.{layer}.att.'
        w=lambda n:state[prefix+n].cuda().bfloat16()
        t=lambda n:trace[f'{layer}.{n}'].cuda().bfloat16()
        n=t('norm1').transpose(0,1)
        delta=torch.cat((torch.zeros_like(n[:1]),n[:-1]),dim=0)-n
        mixed=mix_inputs(n,delta,[w('x_'+ch).flatten() for ch in 'rwkvag'])[5]
        neck=einsum(mixed,w('g1'),'t b d,d r->t b r')
        gate=einsum(torch.sigmoid(neck),w('g2'),'t b r,r d->t b d').transpose(0,1).contiguous()
        ci=[t(name).contiguous() for name in ('groupnorm','core-r','core-k','core-v')]+[w('r_k'),gate]
        assert torch.equal(correction_current(*ci),t('project-input'))
        ni=[t('wkv').contiguous(),w('ln_x.weight'),w('ln_x.bias')]
        assert torch.equal(norm_current(*ni),t('groupnorm'))
        accuracy_rows[str(layer)]={
            'correction':accuracy(ci,funcs['correction'],correction_current,20261010+layer),
            'group_norm':accuracy(ni,funcs['group_norm'],norm_formula,20261011+layer)}
        if layer==15:bench_inputs={'correction':ci,'group_norm':ni}
        print('REAL INPUT CANDIDATES',layer,flush=True)
    benchmarks={name:bench(bench_inputs[name],functions) for name,functions in funcs.items()}
    wkv=wkv_probe(trace,15)
    result={'input':'Current full3B fixed MiniPile seq128; layer0/15/31; key fusion already enabled',
            'accuracy_reference':'Independent FP64 formulas with identical input and upstream gradient',
            'benchmark_input':'Real layer15 tensors, repeated along time for longer lengths; not long-context model outputs',
            'production_changed':False,'accuracy':accuracy_rows,'benchmark':benchmarks,'wkv':wkv}
    (PATH/'fla-candidates-results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
