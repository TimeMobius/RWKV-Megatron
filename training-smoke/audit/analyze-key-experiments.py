"""Summarize exact recorded samples; do not loosen existing tolerances."""
import argparse
import json
from pathlib import Path

import torch
from check import metric

PATH=Path(__file__).resolve().parent


def aggregate(metrics):
    return (sum((m['relative_l2']*m['reference_l2'])**2 for m in metrics.values()) /
            sum(m['reference_l2']**2 for m in metrics.values()))**0.5


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh-fp32',action='store_true')
    options=parser.parse_args()
    results={}
    for path in PATH.glob('full-bf16-fla-vs-blinkdl-main-all-gradients*key-fusion*.json'):
        row=json.loads(path.read_text())
        grads=row['gradient_by_parameter']
        assert len(grads)==1059
        assert all(m['finite'] for m in grads.values())
        results[path.name]={'sequence_length':row['sequence_length'],'token_offset':row['token_offset'],
            'mode':row['precision_probe'] or 'bf16','loss':row['loss_megatron'],'official_loss':row['loss_native'],
            'logits_relative_l2':row['logits']['relative_l2'],
            'v0_relative_l2':row['first_value_gradient']['relative_l2'],
            'aggregate_gradient_relative_l2':aggregate(grads),'parameters':len(grads)}
    fp32_path=PATH/'key-fused-independent-fp32.json'
    if options.refresh_fp32 or not fp32_path.exists():
        fused=torch.load(PATH/'full-key-fused-seq128.pth',map_location='cpu',weights_only=True,mmap=True)
        oracle=torch.load(PATH/'full-fp32-independent-precision-reference.pth',map_location='cpu',weights_only=True,mmap=True)
        metrics={name:metric(value,oracle['gradients'][name]) for name,value in fused['gradients'].items()}
        fp32={'logits':metric(fused['logits'],oracle['logits']),
              'v0_gradient':metric(fused['v0_gradient'],oracle['v0_gradient']),
              'aggregate_gradient_relative_l2':aggregate(metrics),'gradient_by_parameter':metrics,
              'loss_delta':fused['loss']-oracle['loss']}
        fp32_path.write_text(json.dumps(fp32,indent=2)+'\n')
    fp32=json.loads(fp32_path.read_text())
    assert len(fp32['gradient_by_parameter'])==1059
    assert all(m['finite'] for m in fp32['gradient_by_parameter'].values())
    a=torch.load(PATH/'small-key-fused.pth',map_location='cpu',weights_only=True)
    b=torch.load(PATH/'small-key-fused-recompute.pth',map_location='cpu',weights_only=True)
    recompute={'loss_exact':a['loss']==b['loss'],'logits_exact':torch.equal(a['logits'],b['logits']),
               'parameter_count':len(a['gradients']),
               'gradient_exact_count':sum(torch.equal(v,b['gradients'][k]) for k,v in a['gradients'].items())}
    assert recompute['loss_exact'] and recompute['logits_exact']
    assert recompute['gradient_exact_count']==recompute['parameter_count']==135
    operators=json.loads((PATH/'key-operator-experiment.json').read_text())
    for seq,modes in operators['benchmark'].items():
        for name,row in modes.items():
            row['speedup_vs_bf16']=modes['bf16']['median_wall_ms']/row['median_wall_ms']
            row['forward_kernel_count']=len(row['forward_cuda_kernels'])
    benchmark=json.loads((PATH/'key-model-benchmark.json').read_text())
    for seq,modes in benchmark['results'].items():
        for name in ('bf16','key','key-fused'):
            modes[name]['speedup_vs_bf16']=modes['bf16']['median_wall_ms']/modes[name]['median_wall_ms']
    summary={'official_comparisons':results,'fused_vs_independent_fp32_seq128':fp32,
             'recompute':recompute,'operators':operators,'model_benchmark':benchmark,
             'production_default_changed':False,'parallel_or_fp8_certified':False}
    (PATH/'key-experiment-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'official_comparisons':results,'fused_fp32_gradient':fp32['aggregate_gradient_relative_l2'],
                     'recompute':recompute},indent=2))


if __name__=='__main__':main()
