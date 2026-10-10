"""Ensure production fusion matches the tested prototype, including gradients."""
import json
from pathlib import Path

import torch
from check import metric

PATH = Path(__file__).resolve().parent


def main():
    current=torch.load(PATH/'full-production-key-fusion.pth',map_location='cpu',weights_only=True,mmap=True)
    prototype=torch.load(PATH/'full-key-fused-seq128.pth',map_location='cpu',weights_only=True,mmap=True)
    assert current['loss']==prototype['loss']
    assert torch.equal(current['logits'],prototype['logits'])
    assert torch.equal(current['v0_gradient'],prototype['v0_gradient'])
    assert current['gradients'].keys()==prototype['gradients'].keys()
    assert current['trace'].keys()==prototype['trace'].keys()
    exact_grad=sum(torch.equal(v,prototype['gradients'][k]) for k,v in current['gradients'].items())
    exact_trace=sum(torch.equal(v,prototype['trace'][k]) for k,v in current['trace'].items())
    assert exact_grad==len(current['gradients'])==1059
    assert exact_trace==len(current['trace'])==546
    del current,prototype
    a=torch.load(PATH/'small-production-key-fusion.pth',map_location='cpu',weights_only=True)
    b=torch.load(PATH/'small-production-key-fusion-recompute.pth',map_location='cpu',weights_only=True)
    assert a['loss']==b['loss'] and torch.equal(a['logits'],b['logits'])
    recompute_exact=sum(torch.equal(v,b['gradients'][k]) for k,v in a['gradients'].items())
    assert recompute_exact==len(a['gradients'])==135
    eager=torch.load(PATH/'full-production-key-eager.pth',map_location='cpu',weights_only=True,mmap=True)
    oracle=torch.load(PATH/'full-fp32-independent-precision-reference.pth',map_location='cpu',weights_only=True,mmap=True)
    metrics={k:metric(v,oracle['gradients'][k]) for k,v in eager['gradients'].items()}
    assert len(metrics)==1059 and all(m['finite'] for m in metrics.values())
    aggregate=(sum((m['relative_l2']*m['reference_l2'])**2 for m in metrics.values())/
               sum(m['reference_l2']**2 for m in metrics.values()))**0.5
    result={'production_vs_tested_prototype':{'loss_exact':True,'logits_exact':True,'v0_gradient_exact':True,
        'parameter_gradients_exact':exact_grad,'forward_captures_exact':exact_trace},
        'production_recompute':{'loss_exact':True,'logits_exact':True,'parameter_gradients_exact':recompute_exact},
        'fallback_vs_independent_fp32':{'loss_delta':eager['loss']-oracle['loss'],
            'logits':metric(eager['logits'],oracle['logits']),
            'v0_gradient':metric(eager['v0_gradient'],oracle['v0_gradient']),
            'aggregate_gradient_relative_l2':aggregate,'gradient_by_parameter':metrics}}
    (PATH/'production-key-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PRODUCTION == PROTOTYPE:',exact_grad,'gradients;',exact_trace,'forward captures')
    print('RECOMPUTE:',recompute_exact,'gradients exact')
    print('EAGER FP32 aggregate gradient error:',aggregate)


if __name__=='__main__':main()
