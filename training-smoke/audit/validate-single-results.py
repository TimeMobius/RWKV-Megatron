"""Validate frozen single-GPU formula, wiring, recurrence and resume evidence."""
import hashlib
import json
import math
from pathlib import Path

PATH=Path(__file__).resolve().parent


def main():
    fp32=json.loads((PATH/'full-fp32-naive-vs-recurrent-all-gradients.json').read_text())
    assert fp32['gradient_scope']=='all' and len(fp32['gradient_by_parameter'])==1059
    assert abs(fp32['loss_delta'])<1e-5
    assert fp32['logits']['relative_l2']<1e-5
    assert fp32['max_relative_gradient']<1e-3
    assert fp32['first_value_gradient']['relative_l2']<1e-4
    gradients=fp32['gradient_by_parameter'].values()
    relative=math.sqrt(sum((x['relative_l2']*x['reference_l2'])**2 for x in gradients)/
                       sum(x['reference_l2']**2 for x in gradients))
    exact_name='full-bf16-fla-vs-fla-aligned-apex-shared-pointops-te-final-carrier-chain.json'
    bf16=json.loads((PATH/exact_name).read_text())
    assert bf16['loss_delta']==0 and bf16['logits']['max_abs']==0
    assert all(x['max_abs']==0 for x in bf16['trace'].values())
    # Report BF16 gradient differences; do not relabel them bit-exact or change kernel tolerances.
    for x in bf16['gradient_by_parameter'].values():assert x['finite']
    kernel=json.loads((PATH/'kernel-results.json').read_text())
    assert len(kernel)==3 and all(x['passed'] for x in kernel)
    resume=json.loads((PATH/'single-resume-fixed-results.json').read_text())
    assert resume['loss_and_grad_norm_exact_at_logged_precision']
    assert resume['exact_model_optimizer_scheduler'] and resume['rng_exact']
    assert all(not x['differences'] for x in resume['checkpoint_comparisons'].values())
    exits=json.loads((PATH/'single-fixed-exits.json').read_text())
    assert all(x['exit_code']==0 for x in exits.values())
    result={'independent_fp32':{'all_gradients':1059,'loss_delta':fp32['loss_delta'],
            'logits_relative_l2':fp32['logits']['relative_l2'],
            'aggregate_gradient_relative_l2':relative,
            'max_parameter_gradient_relative_l2':fp32['max_relative_gradient'],
            'first_value_gradient_relative_l2':fp32['first_value_gradient']['relative_l2']},
            'bf16_shared_primitives_wiring':{'forward_checkpoints':len(bf16['trace']),
            'logits_bit_exact':True,'loss_exact':True,
            'first_value_gradient_relative_l2':bf16['first_value_gradient']['relative_l2'],
            'max_selected_gradient_relative_l2':bf16['max_relative_gradient'],
            'gradient_scope':'selected; not every full 3B parameter'},
            'resume_model_optimizer_scheduler_rng_exact':True,
            'fla_kernel_reference_sequences':[x['sequence'] for x in kernel],
            'production_change':'Fix pointwise FLA token layout/dispatch',
            'limitations':['One full-model MiniPile prefix, 128 tokens, one sample.',
                           'FP32 formula oracle uses torch recurrence, not FLA.',
                           'BF16 exact-forward diagnostic intentionally shares precision primitives.',
                           'BF16 gradients retain small rounding/accumulation differences.',
                           'Multi-GPU validation paused; previous speed numbers precede this fix.']}
    (PATH/'single-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
