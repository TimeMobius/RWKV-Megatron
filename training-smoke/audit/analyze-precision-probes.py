"""Causal value-sum checks and independent FP32 full-model comparisons."""
import json
import math
from pathlib import Path

import torch

from check import metric

PATH=Path(__file__).resolve().parent
BASE='full-bf16-fla-vs-fla-all-gradients-aligned-shared-pointops-te-final-te-norms-precision-v0-baseline'
PROTO='full-bf16-fla-vs-fla-all-gradients-aligned-shared-pointops-te-final-te-norms-precision-v0-prototype-v0-fp32'


def load(stem):return json.loads((PATH/(stem+'.json')).read_text())
def capture(stem):return torch.load(PATH/(stem+'.pth'),weights_only=True,map_location='cpu',mmap=True)


def main():
    baseline=capture('full-v0-bf16-baseline')
    prototype=capture('full-v0-fp32-prototype')
    a=capture(BASE+'-v0-edges')['actual']
    b=capture(PROTO+'-v0-edges')['actual']
    parts=lambda case,kind:[r['gradient'] for r in case['records'] if r['kind']==kind]
    pa,pb=parts(a,'source-edge'),parts(b,'source-edge')
    la,lb=parts(a,'lerp-end'),parts(b,'lerp-end')
    assert len(la)==len(lb)==31 and len(pa)==len(pb)==3
    assert all(torch.equal(x,y) for x,y in zip(la,lb))
    assert all(torch.equal(x,y) for x,y in zip(pa[1:],pb[1:]))
    exact=sum(t.double() for t in la+pa[1:])
    assert torch.equal(prototype['v0_gradient'].transpose(0,1),exact.to(torch.bfloat16))
    assert baseline['loss']==prototype['loss'] and torch.equal(baseline['logits'],prototype['logits'])
    assert baseline['trace'].keys()==prototype['trace'].keys()
    assert all(torch.equal(t,prototype['trace'][k]) for k,t in baseline['trace'].items())
    changed=[k for k,t in baseline['gradients'].items() if not torch.equal(t,prototype['gradients'][k])]
    allowed={'emb.weight','blocks.0.att.value.weight','blocks.0.att.x_v',
             'blocks.0.ln0.weight','blocks.0.ln0.bias','blocks.0.ln1.weight','blocks.0.ln1.bias'}
    assert set(changed)==allowed,(changed,'unexpected propagation changes')
    value_sum={'all_forward_captures_exact':len(baseline['trace']),
               'loss_exact':True,'later_layer_gradient_branches_exact':31,
               'local_two_gradient_branches_exact':True,
               'baseline_sum_relative_l2':metric(a['total'],exact)['relative_l2'],
               'fp32_source_sum_relative_l2':metric(b['total'],exact)['relative_l2'],
               'fp32_then_bf16_projection_gradient_relative_l2':metric(prototype['v0_gradient'].transpose(0,1),exact)['relative_l2'],
               'projection_gradient_matches_one_final_bf16_round':True,
               'changed_parameters_only':changed,
               'not_a_measure_of_full_model_error':True}
    result={'v0_real_backward':value_sum}
    oracle=capture('full-fp32-independent-precision-reference')
    cases={
        'production_bf16':baseline,'v0_fp32_only':prototype,
        'key_fp32':capture('full-precision-probe-key'),
        'all_intermediates_and_v0_fp32':capture('full-precision-probe-all-v0-fp32'),
        'official_bf16':capture('full-official-bf16-precision-reference'),
    }
    comparisons={}
    for name,case in cases.items():
        assert case['gradients'].keys()==oracle['gradients'].keys()
        gradients={k:metric(v,oracle['gradients'][k]) for k,v in case['gradients'].items()}
        assert all(m['finite'] for m in gradients.values())
        aggregate=math.sqrt(sum((m['relative_l2']*m['reference_l2'])**2 for m in gradients.values())/
                            sum(m['reference_l2']**2 for m in gradients.values()))
        comparisons[name]={'loss':case['loss'],'loss_delta_vs_fp32':case['loss']-oracle['loss'],
            'logits':metric(case['logits'],oracle['logits']),
            'v0_gradient':metric(case['v0_gradient'],oracle['v0_gradient']),
            'all_parameter_gradient_relative_l2':aggregate,
            'parameter_gradients':len(gradients),'gradient_by_parameter':gradients}
        assert len(gradients)==1059
        print(name,'v0',comparisons[name]['v0_gradient']['relative_l2'],'all',aggregate,flush=True)
        result['vs_independent_fp32']=comparisons
        (PATH/'precision-diagnosis-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (PATH/'precision-diagnosis-summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
