"""Gate the updated FLA against independent oracles, and report compatibility differences."""
import json
import math
from pathlib import Path

PATH = Path(__file__).resolve().parent


def read(name):
    return json.loads((PATH / name).read_text())


def main():
    fp32 = read('full-fp32-naive-vs-recurrent-all-gradients-fla-40850297.json')
    assert len(fp32['gradient_by_parameter']) == 1059 and fp32['gradient_scope'] == 'all'
    assert abs(fp32['loss_delta']) < 1e-5 and fp32['logits']['relative_l2'] < 1e-5
    assert fp32['max_relative_gradient'] < 1e-3 and fp32['first_value_gradient']['relative_l2'] < 1e-4
    assert all(v['finite'] for v in fp32['gradient_by_parameter'].values())
    kernel = read('kernel-results-fla-40850297.json')
    assert len(kernel) == 3 and all(v['passed'] for v in kernel)
    after = read('full-bf16-fla-vs-fla-aligned-shared-pointops-te-final-carrier-chain-te-norms-after-fla-40850297.json')
    assert after['loss_delta'] == 0 and after['logits']['max_abs'] == 0
    assert len(after['trace']) == 546 and all(v['max_abs'] == 0 for v in after['trace'].values())
    assert all(v['finite'] for v in after['gradient_by_parameter'].values())
    comparison = read('fla-update-comparison.json')
    assert comparison['results']['logits']['exact'] and comparison['results']['loss']['exact']
    assert all(v['exact'] for v in comparison['results']['trace'].values())
    assert not comparison['all_exact'], 'Do not lose the recorded backward compatibility difference.'
    pointops = read('fla-pointops-comparison.json')
    def leaves(value):
        if isinstance(value, list):
            for row in value:
                yield from leaves(row)
        elif 'exact' in value:
            yield value
        else:
            for row in value.values():
                yield from leaves(row)
    assert all(v['exact'] for v in leaves(pointops))
    for suffix in ('old', 'new', 'old-pinned', 'new-pinned'):
        assert all(v['passed'] for v in read(f'fla-pointops-{suffix}.json').values())
    diagnostic = read('fla-old-dgk-diagnostic.json')
    assert diagnostic['v0_exact'] and diagnostic['layer31_decay']['exact']
    assert all(v['exact'] == v['total'] for k, v in diagnostic.items() if isinstance(v, dict) and 'total' in v)
    core = read('fla-core-gradient-replay.json')
    assert core['old-repeat']['v0_gradient']['exact'] and core['new-repeat']['v0_gradient']['exact']
    rows = comparison['training_rows']
    assert len(rows) == 4 and all(row[-2:] == ['0', '0'] for row in rows)
    exits = read('fla-update-exits.json')
    for name in ('upstream-rwkv7-fla-40850297', 'single-training-fla-40850297', 'compare-report-fla-40850297'):
        assert exits[name]['exit_code'] == 0
    assert exits['compare-exact-before-diagnosis']['exit_code'] == 1
    log = (PATH.parent / 'logs/audit-upstream-rwkv7-fla-40850297.log').read_text()
    assert '31 passed' in log
    log = (PATH.parent / 'logs/regression-fla-40850297.log').read_text()
    assert '88 passed' in log
    values = list(fp32['gradient_by_parameter'].values())
    aggregate = math.sqrt(sum((v['relative_l2'] * v['reference_l2'])**2 for v in values)
                          / sum(v['reference_l2']**2 for v in values))
    result = dict(old_commit=comparison['before_commit'], new_commit=comparison['after_commit'],
        fast_forward_commits=29, custom_fla_changes=False, root_regression_passed=88,
        upstream_rwkv7_tests_passed=31, kernel_reference_sequences=[64, 128, 129],
        independent_fp32=dict(parameters=1059, loss_delta=fp32['loss_delta'],
            aggregate_gradient_relative_l2=aggregate, max_parameter_gradient_relative_l2=fp32['max_relative_gradient']),
        bf16_reference_forward_exact=True, old_new_546_forward_exact=True,
        old_new_gradients_exact=False, old_new_v0_gradient_relative_l2=comparison['results']['v0_gradient']['relative_l2'],
        max_selected_parameter_old_new_relative_l2=max(v['relative_l2'] for v in comparison['results']['gradients'].values()),
        first_core_difference=core['old-new-replay']['core_gradients']['31.core-decay'],
        diagnosis='Only restoring the old chunk_dplr_bwd_dgk_kernel in a diagnostic process makes all356 selected parameter gradients and all482 backward captures exact again.',
        pointops_128_4096_output_gradient_exact=True, repeat_old_new_v0_exact=True,
        four_step_training_zero_nan_skips=True, four_step_loss_gradient_trajectory_exact=False,
        validation_passed=True, cross_version_bit_exact_gate_passed=False,
        limitations=['Single-GPU 3B prefix128 and four training steps; no long-run convergence claim.',
                    'BF16 cross-version backward/optimizer trajectory changes despite identical initial forward.',
                    'Old-kernel override is diagnostic only; production FLA remains upstream unmodified.',
                    'No new SP/TP/PP/FP8 or checkpoint-resume certification after this FLA update.',
                    'No throughput improvement claimed.'])
    (PATH / 'fla-update-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
