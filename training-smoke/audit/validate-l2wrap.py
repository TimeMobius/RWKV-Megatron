"""Validate unchanged CE mode, official loss, model gradients and real training."""
import json
import re
from pathlib import Path

import torch

PATH=Path(__file__).resolve().parent
ROOT=PATH.parent


def main():
    current=torch.load(PATH/'full-l2wrap-disabled.pth',map_location='cpu',weights_only=True,mmap=True)
    previous=torch.load(PATH/'full-production-key-fusion.pth',map_location='cpu',weights_only=True,mmap=True)
    assert current['loss']==previous['loss'] and torch.equal(current['logits'],previous['logits'])
    exact_grad=sum(torch.equal(v,previous['gradients'][k]) for k,v in current['gradients'].items())
    exact_trace=sum(torch.equal(v,previous['trace'][k]) for k,v in current['trace'].items())
    assert exact_grad==len(current['gradients'])==1059
    assert exact_trace==len(current['trace'])==546
    del current,previous
    resume=json.loads((PATH/'single-resume-l2wrap-results.json').read_text())
    assert resume['exact_model_optimizer_scheduler'] and resume['rng_exact']
    assert all(not c['differences'] for c in resume['checkpoint_comparisons'].values())
    training=json.loads((PATH/'l2wrap-training-exits.json').read_text())
    assert all(row['exit_code']==0 for row in training.values())
    official=json.loads((PATH/'l2wrap-official-results.json').read_text())
    tp=json.loads((PATH/'l2wrap-tp-results.json').read_text())
    model=json.loads((PATH/'l2wrap-model-results.json').read_text())
    assert tp['mask_and_cross_shard_tie_passed']
    assert len(model['results'])==3
    assert all(row['forward_ce_exact'] for row in model['results'].values())
    units=(ROOT/'logs/audit-l2wrap-unit.log').read_text()
    assert '91 passed' in units
    smoke=(ROOT/'logs/l2wrap-per-token-key-eager.log').read_text()
    for pattern in (r'calculate_per_token_loss\s+\.+ True',r'rwkv_key_fusion\s+\.+ False',
                    r'rwkv_l2wrap_coefficient\s+\.+ 0.0001'):
        assert re.search(pattern,smoke),pattern
    iterations=re.findall(r'iteration\s+(\d+)/\s*2.*?lm loss:\s*([\d.Ee+-]+).*?grad norm:\s*([\d.Ee+-]+).*?skipped iterations:\s*(\d+).*?nan iterations:\s*(\d+)',smoke)
    assert [row[0] for row in iterations]==['1','2']
    assert all(row[-2:]==('0','0') for row in iterations)
    result={'disabled_mode':{'loss_exact':True,'logits_exact':True,
                            'parameter_gradients_exact':exact_grad,'forward_captures_exact':exact_trace},
            'unit_tests_passed':91,'subtests_passed':30,'official_cuda_loss':official,
            'real_tp_loss':tp,'whole_small_model_gradient':model,
            'full_3b_resume':resume,'full_3b_per_token_eager_smoke':{
                'key_fusion_yaml':False,'coefficient':1e-4,'per_token_loss':True,
                'microbatches':2,'iterations':iterations},
            'multi_gpu_rwkv_model_or_fp8_certified':False}
    (PATH/'l2wrap-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('L2WRAP VALIDATION PASS: disabled mode exact, official gradient, TP loss, model gradient, full3B resume and eager per-token training')


if __name__=='__main__':main()
