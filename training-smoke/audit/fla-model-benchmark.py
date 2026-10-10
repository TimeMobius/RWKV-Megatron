"""Full3B timing of process-local FLA candidates on actual MiniPile sequences."""
import gc
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars

import distributed as audit
from key_experiment_import import timed
from fla_candidate_probe import install
from rwkv_megatron.rwkv7_model import RWKV7Model
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec

PATH=Path(__file__).resolve().parent


def main():
    torch.cuda.set_device(0);dist.init_process_group('nccl')
    ps.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234);torch.backends.cuda.matmul.allow_tf32=False
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    audit.FULL=True;audit.DIMS={'layers':32,'hidden':2560,'heads':40,'ffn':10240,'vocab':65536,
                              'sequence':2048,'batch':1,'microbatches':1}
    cfg=audit.config_for(1,1,False,False)
    model=RWKV7Model(cfg,transformer_layer_spec=get_rwkv7_layer_spec(),vocab_size=65536,
        max_sequence_length=2048,pre_process=True,post_process=True,parallel_output=False,
        share_embeddings_and_output_weights=False,position_embedding_type='none').cuda().bfloat16()
    state=torch.load('/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth',map_location='cpu',weights_only=True)
    model.from_rwkv_lm(state,False);del state;gc.collect();model.train()
    modules=[m for layer in model.decoder.layers for m in (layer.self_attention,
            layer.self_attention.group_norm,layer.self_attention.core_attention)]
    originals=[(module,module.forward) for module in modules]
    tokens=torch.load(PATH/'key-minipile-4097.pth',weights_only=True).cuda()
    results={};names=['current','correction','norm','norm-correction','wkv-save']
    for seq in (512,2048):
        cfg.recompute_granularity='full' if seq==2048 else None
        cfg.recompute_method='uniform' if seq==2048 else None
        cfg.recompute_num_layers=1 if seq==2048 else None
        x,y=tokens[:,:seq],tokens[:,1:seq+1];position=torch.arange(seq,device=x.device).expand_as(x)
        def step():
            model.zero_grad(set_to_none=True)
            out=model(x,position,None)
            loss=F.cross_entropy(out.float().reshape(-1,65536),y.reshape(-1))
            loss.backward();return loss
        samples={n:[] for n in names};peaks={};losses={};warmed=set()
        for round_index in range(4):
            order=names[round_index%len(names):]+names[:round_index%len(names)]
            for name in order:
                for module,forward in originals:
                    module.forward=forward
                    if hasattr(module,'_audit_forward_source'):del module._audit_forward_source
                if name!='current':install(model,name)
                if name not in warmed:
                    for _ in range(3):step()
                    warmed.add(name)
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                samples[name].append(timed(step,4))
                peaks[name]=max(peaks.get(name,0),torch.cuda.max_memory_allocated())
                loss=step();assert torch.isfinite(loss);losses[name]=loss.item()
                print('FULL FLA BENCH',seq,round_index,name,samples[name][-1],flush=True)
        results[str(seq)]={n:{'samples':samples[n],'median_wall_ms':statistics.median(s['wall_ms'] for s in samples[n]),
            'tokens_per_second':seq*1000/statistics.median(s['wall_ms'] for s in samples[n]),
            'peak_allocated_bytes':peaks[n],'loss':losses[n]} for n in names}
        for n in names:results[str(seq)][n]['speedup_vs_current']=results[str(seq)]['current']['median_wall_ms']/results[str(seq)][n]['median_wall_ms']
        (PATH/'fla-model-benchmark-results.json').write_text(json.dumps({'results':results,
            'scope':'Full3B BF16 batch1 forward+plain CE+backward+zero_grad; no optimizer/data IO/checkpoint IO; production source unchanged',
            'recompute':'none at seq512; full per-layer at seq2048','timing':'Warmup excluded; five modes rotated over four rounds of four iterations'},indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
