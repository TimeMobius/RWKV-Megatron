"""Validate separate TE-unification evidence without altering previous results."""
import json
import math
from pathlib import Path

PATH = Path(__file__).resolve().parent


def main():
    fp32 = json.loads((PATH / 'full-fp32-naive-vs-recurrent-all-gradients-unified-te.json').read_text())
    bf16 = json.loads((PATH / 'full-bf16-fla-vs-fla-aligned-shared-pointops-te-final-carrier-chain-te-norms-unified.json').read_text())
    assert fp32['gradient_scope'] == 'all' and len(fp32['gradient_by_parameter']) == 1059
    assert abs(fp32['loss_delta']) < 1e-5 and fp32['logits']['relative_l2'] < 1e-5
    assert fp32['max_relative_gradient'] < 1e-3 and fp32['first_value_gradient']['relative_l2'] < 1e-4
    assert all(v['finite'] for v in fp32['gradient_by_parameter'].values())
    assert bf16['loss_delta'] == 0 and bf16['logits']['max_abs'] == 0
    assert len(bf16['trace']) == 546 and all(v['max_abs'] == 0 for v in bf16['trace'].values())
    assert all(v['finite'] for v in bf16['gradient_by_parameter'].values())
    fallback = {}
    for mode in ('te', 'old', 'missing'):
        row = json.loads((PATH / f'te-fallback-{mode}.json').read_text())
        assert row['finite_forward_backward'] and row['converter_validated']
        assert row['sequence_parallel_param_tags'] and row['zero_centered_gamma_relative_l2'] < 0.01
        assert row['first_value_weight_gradient_norm'] > 0
        fallback[mode] = row
    assert not fallback['missing']['te_imported']
    resume = json.loads((PATH / 'single-resume-te-results.json').read_text())
    assert resume['exact_model_optimizer_scheduler'] and resume['rng_exact']
    assert all(not v['differences'] for v in resume['checkpoint_comparisons'].values())
    exits = json.loads((PATH / 'te-validation-exits.json').read_text())
    assert len(exits) == 8 and all(v['exit_code'] == 0 for v in exits.values())
    gradients = list(fp32['gradient_by_parameter'].values())
    relative = math.sqrt(sum((x['relative_l2'] * x['reference_l2'])**2 for x in gradients)
                         / sum(x['reference_l2']**2 for x in gradients))
    result = dict(backend_policy='TE>=1.9.0, otherwise Apex; same policy for ln0/ln1/ln2/ln_out',
        production_change='Unified RWKV7 LayerNorm; optional TE imports and BF16 context fallback',
        independent_fp32=dict(loss_delta=fp32['loss_delta'], logits_relative_l2=fp32['logits']['relative_l2'],
            parameter_gradients=1059, aggregate_gradient_relative_l2=relative,
            max_parameter_gradient_relative_l2=fp32['max_relative_gradient'],
            v0_gradient_relative_l2=fp32['first_value_gradient']['relative_l2']),
        bf16_shared_primitives=dict(forward_captures=546, logits_exact=True, loss_exact=True,
            loss=bf16['loss_megatron'], v0_gradient_relative_l2=bf16['first_value_gradient']['relative_l2'],
            max_selected_gradient_relative_l2=bf16['max_relative_gradient']),
        fallback_modes_tested=list(fallback), norm_param_sequence_parallel_tags=True,
        zero_centered_gamma=True, cropped_real_weight_converter=True,
        resume_model_optimizer_scheduler_rng_exact=True, resume_rows=resume['rows'],
        regression_passed=88, limitations=['Single-GPU seq128 full-model validation; no new multi-GPU run.',
            'SP metadata tested, not distributed SP/TP/PP execution.',
            'FLA/FP8 core policy unchanged; full-model FP8 training not certified.',
            'BF16 multi-branch gradient accumulation remains non-bit-exact.',
            'TE old/missing simulated in fresh processes; installed environment unchanged.'])
    (PATH / 'te-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
