"""Full-model candidate accuracy against independent FP32 and real timings."""
import json
from pathlib import Path

import torch
from check import metric

PATH=Path(__file__).resolve().parent


def aggregate(metrics):
    return (sum((m['relative_l2']*m['reference_l2'])**2 for m in metrics.values())/
            sum(m['reference_l2']**2 for m in metrics.values()))**.5


def main():
    oracle=torch.load(PATH/'full-fp32-independent-precision-reference.pth',map_location='cpu',weights_only=True,mmap=True)
    baseline=torch.load(PATH/'full-production-key-fusion.pth',map_location='cpu',weights_only=True,mmap=True)
    accuracy={}
    for mode in ('current','correction','norm','norm-correction','wkv-save'):
        cap=baseline if mode=='current' else torch.load(PATH/f'full-fla-candidate-{mode}.pth',map_location='cpu',weights_only=True,mmap=True)
        metrics={k:metric(v,oracle['gradients'][k]) for k,v in cap['gradients'].items()}
        assert len(metrics)==1059 and all(m['finite'] for m in metrics.values())
        row={'logits':metric(cap['logits'],oracle['logits']),
             'loss':cap['loss'],'loss_delta_vs_fp32':cap['loss']-oracle['loss'],
             'v0_gradient':metric(cap['v0_gradient'],oracle['v0_gradient']),
             'aggregate_parameter_gradient_relative_l2':aggregate(metrics),
             'gradient_by_parameter':metrics,
             'vs_current':{'loss_exact':cap['loss']==baseline['loss'],
                           'logits_exact':torch.equal(cap['logits'],baseline['logits']),
                           'v0_gradient_exact':torch.equal(cap['v0_gradient'],baseline['v0_gradient']),
                           'parameter_gradients_exact':sum(torch.equal(v,baseline['gradients'][k]) for k,v in cap['gradients'].items()),
                           'forward_captures_exact':sum(torch.equal(v,baseline['trace'][k]) for k,v in cap['trace'].items())}}
        if mode=='wkv-save':
            assert row['vs_current']['loss_exact'] and row['vs_current']['logits_exact']
            assert row['vs_current']['parameter_gradients_exact']==1059
        accuracy[mode]=row
        print('FULL CANDIDATE ACCURACY',mode,'aggregate gradient',row['aggregate_parameter_gradient_relative_l2'],
              'logits',row['logits']['relative_l2'],'v0',row['v0_gradient']['relative_l2'],flush=True)
    local=json.loads((PATH/'fla-candidates-results.json').read_text())
    timing=json.loads((PATH/'fla-model-benchmark-results.json').read_text())
    exits=json.loads((PATH/'fla-candidate-exits.json').read_text())
    assert all(row['exit_code']==0 for row in exits.values())
    result={'full_model_vs_independent_fp32':accuracy,'local_operators':local,'model_benchmark':timing,
            'production_changed':False,'environment_changed':False,'fla_source_changed':False,
            'tilelang_installed':False,'multi_gpu_or_fp8_certified':False}
    (PATH/'fla-candidate-summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
