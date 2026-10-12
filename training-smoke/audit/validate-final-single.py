"""Validate current single-GPU evidence without declaring BF16 bit equality."""

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'training-smoke/audit'


def read(name):
    return json.loads((AUDIT / name).read_text())


def summary(case):
    gradients = case['gradient_by_parameter']
    assert case['full_3b'] and case['gradient_scope'] == 'all' and len(gradients) == 1059
    assert all(value['finite'] for value in gradients.values())
    assert case['logits']['finite'] and case['first_value_gradient']['finite']
    return {
        'loss_megatron': case['loss_megatron'], 'loss_reference': case['loss_native'],
        'loss_delta': case['loss_delta'], 'logits_relative_l2': case['logits']['relative_l2'],
        'aggregate_gradient_relative_l2': math.sqrt(
            sum((v['relative_l2'] * v['reference_l2']) ** 2 for v in gradients.values())
            / sum(v['reference_l2'] ** 2 for v in gradients.values())),
        'v0_gradient_relative_l2': case['first_value_gradient']['relative_l2'],
        'max_parameter_gradient_relative_l2': case['max_relative_gradient'],
        'worst_parameter': max(gradients, key=lambda key: gradients[key]['relative_l2']),
        'parameter_gradients': len(gradients),
        'exact_parameter_gradients': sum(v['max_abs'] == 0 for v in gradients.values()),
        'forward_captures': len(case['trace']),
        'exact_forward_captures': sum(v['max_abs'] == 0 for v in case['trace'].values()),
    }


def main():
    files = {
        'fp32_formula': 'full-fp32-naive-vs-blinkdl-simplified-all-gradients-final-20261012-blinkdl-source.json',
        'bf16_official': 'full-bf16-fla-vs-blinkdl-main-all-gradients-final-20261012-blinkdl-source.json',
        'direct_vfirst_diagnostic': 'full-bf16-fla-vs-fla-all-gradients-aligned-shared-pointops-te-final-te-norms-final-direct-vfirst-20261012.json',
    }
    cases = {name: read(file) for name, file in files.items()}
    fp32 = cases['fp32_formula']
    # Frozen formula thresholds from validate-blinkdl-alignment.py.
    assert abs(fp32['loss_delta']) < 1e-5
    assert fp32['logits']['relative_l2'] < 1e-5
    assert fp32['max_relative_gradient'] < 1e-3
    assert fp32['first_value_gradient']['relative_l2'] < 1e-4
    for name in ('fp32_formula', 'bf16_official'):
        provenance = cases[name]['reference_provenance']
        assert provenance['exact_source_forward_bodies']
        assert provenance['v_first_transport'] == 'direct object reuse, no carrier'
        for file, sha in provenance['source_sha256'].items():
            source = AUDIT.parent / 'reference-repos/RWKV-LM/RWKV-v7/train_temp' / file
            assert hashlib.sha256(source.read_bytes()).hexdigest() == sha
    diagnostic = cases['direct_vfirst_diagnostic']
    assert diagnostic['production_key_reference'] and diagnostic['reference_profile'] == 'diagnostic'
    assert '-carrier-chain' not in files['direct_vfirst_diagnostic']
    assert diagnostic['loss_delta'] == diagnostic['logits']['max_abs'] == diagnostic['first_value_gradient']['max_abs'] == 0
    assert all(v['max_abs'] == 0 for v in diagnostic['gradient_by_parameter'].values())
    assert all(v['max_abs'] == 0 for v in diagnostic['trace'].values())
    previous = read('full-bf16-fla-vs-blinkdl-main-all-gradients-production-key-fusion-blinkdl-source.json')
    bf16 = cases['bf16_official']
    assert bf16['loss_megatron'] == previous['loss_megatron']
    assert bf16['logits'] == previous['logits']
    assert bf16['first_value_gradient'] == previous['first_value_gradient']
    for name, metrics in bf16['gradient_by_parameter'].items():
        # Official CUDA coefficient-gradient reductions vary slightly between
        # runs. Check unchanged production gradient norms and maximum errors;
        # preserve all measured reference differences without relaxing kernel
        # or independent FP32 formula thresholds.
        assert metrics['actual_l2'] == previous['gradient_by_parameter'][name]['actual_l2'], name
        assert metrics['max_abs'] == previous['gradient_by_parameter'][name]['max_abs'], name
    kernels = {name: read(f'final-single-20261012/{name}')
               for name in ('kernel-results.json', 'blinkdl-kernel-results.json')}
    assert [r['sequence'] for r in kernels['kernel-results.json']] == [64, 128, 129]
    assert all(row['passed'] for rows in kernels.values() for row in rows)
    assert all(row['thresholds'] == {'output': 0.007, 'gradients': 0.008}
               for row in kernels['blinkdl-kernel-results.json'])
    tests_log = (ROOT / 'training-smoke/logs/final-single-tests-fixed-20261012.log').read_text()
    assert '133 passed, 2 xfailed' in tests_log and '30 subtests passed' in tests_log
    log = (ROOT / 'training-smoke/logs/final-3b-seq2048-20261012.log').read_text()
    assert re.search(r'\n\s+seq_length\s+\.+\s+2048', log)
    assert '[after training is done]' in log and 'successfully saved checkpoint' not in log
    pattern = re.compile(r'\] iteration\s+(\d+)/\s*\d+.*?consumed samples:\s*(\d+)'
                         r'.*?learning rate:\s*([\d.Ee+-]+).*?lm loss:\s*([\d.Ee+-]+).*?grad norm:\s*([\d.Ee+-]+)'
                         r'.*?skipped iterations:\s*(\d+).*?nan iterations:\s*(\d+)')
    rows = []
    for match in pattern.finditer(log):
        step, samples, lr, loss, grad, skipped, nan = match.groups()
        rows.append({'step': int(step), 'samples': int(samples), 'lr': float(lr), 'loss': float(loss),
                     'grad_norm': float(grad), 'skipped': int(skipped), 'nan': int(nan)})
    assert [r['step'] for r in rows] == [1, 2]
    assert all(r['nan'] == r['skipped'] == 0 and math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows)
    evals = [float(x) for x in re.findall(r'validation loss at iteration.*?lm loss value: ([\d.Ee+-]+)', log)]
    assert len(evals) == 3 and all(math.isfinite(v) for v in evals) and evals[-1] == evals[-2]
    resumes = {}
    for name in ('single-resume-fixed-eval-results.json', 'single-resume-muon-20261010-results.json'):
        result = read(name)
        assert result['exact_model_optimizer_scheduler'] and result['rng_exact']
        assert all(not row['differences'] for row in result['checkpoint_comparisons'].values())
        resumes[name] = {'recorded_exact_model_optimizer_scheduler_rng': True,
                         'source': name, 'rerun_on_20261012': False}
    result = {
        'date': '2026-10-12', 'workspace': str(ROOT),
        'scope': 'Single H800; full original 3B; BF16; TP1/PP1/SP off',
        'cases': {name: summary(case) for name, case in cases.items()},
        'case_files': files, 'formula_thresholds_passed': True,
        'v0_wiring_all_gradients_exact_with_shared_primitives': True,
        'bf16_precision_matches_prior_production_result': True,
        'bf16_equivalence_to_official_certified': False,
        'kernel_frozen_thresholds_passed': True,
        'tests': {'passed': 133, 'subtests_passed': 30, 'historical_negative_controls_xfailed': 2},
        'seq2048_production_training': {'steps': rows, 'eval_losses': evals, 'checkpoint_files_written': False},
        'previous_resume_evidence_reviewed': resumes,
        'runtime': bf16['runtime'],
        'source_sha256': {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
                          for file in sorted((ROOT / 'rwkv_megatron').glob('*.py'))},
        'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'limitations': ['No long-run convergence or universal BF16 error bound established.',
                        'FP8 remains experimental; existing precision differences are material.',
                        'Muon torch checkpoint uses use_layer_wise_param_layout=false.',
                        'Full multi-GPU TP/SP/PP/DP and topology-changing resume not certified.',
                        'No new checkpoint files generated; earlier raw checkpoints were deleted after verification.'],
    }
    (AUDIT / 'final-single-validation-20261012.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('runtime', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
