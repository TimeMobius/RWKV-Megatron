"""Single GPU key accuracy, fusion, and forward/backward throughput experiment."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from check import metric
from key_precision import key_compiled
from operator_precision_import import key_current, key_formula

PATH = Path(__file__).resolve().parent
MODEL = '/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth'


def key_high(k, a, beta, alpha):
    from rwkv_megatron.rwkv7_layer_specs import update_key
    kk = F.normalize((k.float()*beta.float()).reshape(*k.shape[:2],-1,64),dim=-1).flatten(2)
    return update_key(k.float(),a.float(),alpha.float()).to(k.dtype), (-kk).to(k.dtype), (a.float()*kk).to(k.dtype)


def prepare_tokens():
    import numpy as np
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    provenance=json.loads((PATH/'minipile-input.json').read_text())
    prefix=provenance['cache_prefix']
    doc=np.load(prefix+'-document_index.npy',mmap_mode='r')
    sample=np.load(prefix+'-sample_index.npy',mmap_mode='r')
    index,offset=sample[provenance['sample_index']]
    data=IndexedDataset('/mnt/data/Datasets/minipile/minipile_text_document')
    chunks=[];remaining=4097
    while remaining:
        document=int(doc[index]);length=min(remaining,int(data.sequence_lengths[document])-int(offset))
        chunks.append(data.get(document,offset=int(offset),length=length).astype(np.int64))
        remaining-=length;index+=1;offset=0
    tokens=torch.from_numpy(np.concatenate(chunks)).unsqueeze(0)
    original=torch.load(PATH/'minipile-tokens.pth',weights_only=True)
    assert torch.equal(tokens[:,:original.shape[1]],original)
    torch.save(tokens,PATH/'key-minipile-4097.pth')
    provenance.pop('token_ids',None)
    (PATH/'key-minipile-input.json').write_text(json.dumps({**provenance,'sequence_length':4096,
        'token_count':tokens.numel(),'token_file':'key-minipile-4097.pth',
        'original_prefix_exact':True},indent=2)+'\n')
    print('TOKEN COPY READY',tokens.shape,flush=True)


def timed(step, repeats):
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    wall=time.perf_counter();start.record()
    for _ in range(repeats):step()
    end.record();torch.cuda.synchronize()
    return {'gpu_ms':start.elapsed_time(end)/repeats,
            'wall_ms':1000*(time.perf_counter()-wall)/repeats}


def operators(skip_benchmark=False):
    from einops import einsum
    from rwkv_megatron.rwkv7_layer_specs import mix_inputs
    torch.manual_seed(20261010)
    state=torch.load(MODEL,map_location='cpu',weights_only=True)
    capture=torch.load(PATH/'full-new-replay-core.pth',map_location='cpu',weights_only=True)
    trace=capture['trace'];rows={};bench={}
    if skip_benchmark:
        bench=json.loads((PATH/'key-operator-experiment.json').read_text())['benchmark']
    funcs={'bf16':key_current,'fp32':key_high,'fp32_fused':key_compiled}
    for layer in (0,15,31):
        prefix=f'blocks.{layer}.att.'
        w=lambda name:state[prefix+name].cuda().to(torch.bfloat16)
        n=trace[f'{layer}.norm1'].cuda().transpose(0,1).to(torch.bfloat16)
        delta=torch.cat((torch.zeros_like(n[:1]),n[:-1]),dim=0)-n
        mixed=mix_inputs(n,delta,[w('x_'+ch).flatten() for ch in 'rwkvag'])[4]
        neck=einsum(mixed,w('a1'),'t b d, d r -> t b r')
        pre=einsum(neck,w('a2'),'t b r, r d -> t b d')
        learn=torch.sigmoid(pre+w('a0'))
        inputs=[trace[f'{layer}.k-raw'].cuda().transpose(0,1).to(torch.bfloat16),learn,
                w('k_k').flatten(),w('k_a').flatten()]
        oracle=[t.double().detach().requires_grad_() for t in inputs]
        expected=key_formula(*oracle)
        # Accuracy inputs must stay fixed whether timing is requested or skipped.
        torch.manual_seed(20261010+layer)
        dy=[torch.randn_like(t).to(torch.bfloat16) for t in expected]
        refgrad=torch.autograd.grad(expected,oracle,[t.double() for t in dy])
        row={}
        for name,fn in funcs.items():
            tensors=[t.detach().requires_grad_() for t in inputs]
            out=fn(*tensors)
            grad=torch.autograd.grad(out,tensors,dy)
            row[name]={'output':[metric(a,b) for a,b in zip(out,expected)],
                       'gradient':[metric(a,b) for a,b in zip(grad,refgrad)]}
            assert all(m['finite'] for group in row[name].values() for m in group)
        float_inputs=[t.detach().float().requires_grad_() for t in inputs]
        float_out=key_compiled(*float_inputs)
        float_grad=torch.autograd.grad(float_out,float_inputs,[t.float() for t in dy])
        row['compiled_fp32_formula']={'output':[metric(a,b) for a,b in zip(float_out,expected)],
                                     'gradient':[metric(a,b) for a,b in zip(float_grad,refgrad)]}
        assert max(m['relative_l2'] for group in row['compiled_fp32_formula'].values() for m in group)<1e-5
        # Identical real inputs and upstream gradients, not different models.
        assert max(m['relative_l2'] for m in row['fp32_fused']['gradient']) < max(m['relative_l2'] for m in row['bf16']['gradient'])
        baseline=key_current(*inputs)
        for output,name in zip(baseline,('core-k','core-a','core-b')):
            assert torch.equal(output.transpose(0,1).cpu(),trace[f'{layer}.{name}'])
        rows[str(layer)]=row
        if layer==15 and not skip_benchmark:
            for seq in (128,512,2048):
                bench[str(seq)]={}
                for name,fn in funcs.items():
                    tensors=[(t.repeat(seq//128,1,1) if t.ndim==3 else t).detach().requires_grad_() for t in inputs]
                    upstream=[torch.randn_like(tensors[0]) for _ in range(3)]
                    step=lambda:torch.autograd.grad(fn(*tensors),tensors,upstream)
                    for _ in range(3):step()
                    samples=[timed(step,30) for _ in range(5)]
                    input_storages={t.untyped_storage().data_ptr() for t in tensors}
                    saved={}
                    def pack(t):
                        storage=t.untyped_storage()
                        if storage.data_ptr() not in input_storages:saved[storage.data_ptr()]=storage.nbytes()
                        return t
                    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
                        out=fn(*tensors)
                    torch.autograd.grad(out,tensors,upstream)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as p:
                        out=fn(*tensors)
                    kernels=[e.name for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                    bench[str(seq)][name]={'samples':samples,'median_wall_ms':statistics.median(s['wall_ms'] for s in samples),
                        'extra_saved_bytes':sum(saved.values()),'forward_cuda_kernels':kernels}
                print('OPERATOR BENCH',seq,flush=True)
        print('KEY ACCURACY',layer,flush=True)
    (PATH/'key-operator-experiment.json').write_text(json.dumps({'accuracy':rows,'benchmark':bench,
        'benchmark_reused_for_additional_formula_check':skip_benchmark,
        'oracle':'Independent FP64 formula; BF16 inputs and identical upstream gradients',
        'benchmark_inputs':'Real layer15 inputs repeated for longer lengths'},indent=2)+'\n')


def model_benchmark():
    import distributed as audit
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.training import global_vars
    from rwkv_megatron.rwkv7_model import RWKV7Model
    from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
    from precision_probe import install
    ps.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    audit.FULL=True;audit.DIMS={'layers':32,'hidden':2560,'heads':40,'ffn':10240,'vocab':65536,
                              'sequence':2048,'batch':1,'microbatches':1}
    cfg=audit.config_for(1,1,False,False)
    model=RWKV7Model(cfg,transformer_layer_spec=get_rwkv7_layer_spec(),vocab_size=65536,
                    max_sequence_length=2048,pre_process=True,post_process=True,parallel_output=False,
                    share_embeddings_and_output_weights=False,position_embedding_type='none').cuda().bfloat16()
    state=torch.load(MODEL,map_location='cpu',weights_only=True)
    model.from_rwkv_lm(state,False);del state;gc.collect();model.train()
    originals=[(layer.self_attention,layer.self_attention.forward) for layer in model.decoder.layers]
    tokens=torch.load(PATH/'key-minipile-4097.pth',weights_only=True).cuda()
    rows={}
    for seq in (128,512,2048):
        cfg.recompute_granularity='full' if seq==2048 else None
        cfg.recompute_method='uniform' if seq==2048 else None
        cfg.recompute_num_layers=1 if seq==2048 else None
        x,y=tokens[:,:seq],tokens[:,1:seq+1]
        positions=torch.arange(seq,device=x.device).expand_as(x)
        def step():
            model.zero_grad(set_to_none=True)
            logits=model(x,positions,None)
            loss=F.cross_entropy(logits.float().reshape(-1,65536),y.reshape(-1))
            loss.backward()
            return loss
        samples={name:[] for name in ('bf16','key','key-fused')}
        peaks={};losses={};warmed=set()
        for round_index in range(4):
            order=list(samples)
            order=order[round_index%3:]+order[:round_index%3]
            for name in order:
                for att,forward in originals:
                    att.forward=forward
                    if hasattr(att,'_audit_forward_source'):del att._audit_forward_source
                if name!='bf16':install(model,name)
                if name not in warmed:
                    for _ in range(3):step()
                    warmed.add(name)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                samples[name].append(timed(step,4))
                peaks[name]=max(peaks.get(name,0),torch.cuda.max_memory_allocated())
                loss=step();losses[name]=loss.item()
                assert torch.isfinite(loss)
                print('MODEL BENCH',seq,round_index,name,samples[name][-1],flush=True)
        rows[str(seq)]={name:{'samples':data,'median_wall_ms':statistics.median(s['wall_ms'] for s in data),
            'tokens_per_second':seq*1000/statistics.median(s['wall_ms'] for s in data),
            'peak_allocated_bytes':peaks[name],'loss':losses[name]} for name,data in samples.items()}
        rows[str(seq)]['recompute']=seq==2048
        (PATH/'key-model-benchmark.json').write_text(json.dumps({'results':rows,
            'scope':'Full 3B batch1 forward+plain CE+backward+zero_grad; no optimizer/data IO/checkpoint IO',
            'timing':'Warmup excluded, modes rotated over four rounds of four iterations'},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('tokens','operator','model'))
    parser.add_argument('--skip-benchmark',action='store_true')
    args=parser.parse_args()
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl')
    if args.mode=='tokens':prepare_tokens()
    elif args.mode=='operator':operators(args.skip_benchmark)
    else:model_benchmark()
    dist.destroy_process_group()


if __name__=='__main__':main()
