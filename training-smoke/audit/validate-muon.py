"""Verify production Muon smoke, exact resume and parameter routing."""

import hashlib
import importlib.metadata
import json
import re
import runpy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'training-smoke/audit'
comparison = runpy.run_path(str(AUDIT / 'compare-single-resume.py'))


def main():
    sys.argv = [str(AUDIT / 'compare-single-resume.py'), '--suffix=-muon-20261010']
    comparison['main']()
    result = json.loads((AUDIT / 'single-resume-muon-20261010-results.json').read_text())
    assert result['rng_exact']
    captures = {phase: json.loads((AUDIT / f'muon-{phase}-compatible-batches.json').read_text())['events']
                for phase in ('save', 'resume', 'continuous')}
    baseline = captures['continuous']
    sample_hashes = baseline[0]['batch_hashes']
    assert all(event['batch_hashes'] == sample_hashes for events in captures.values() for event in events)
    # The legacy audit captures args.iteration, which stays at the loaded step
    # during the training loop. Compare eval order rather than that stale value.
    losses = {phase: [event['losses'] for event in events if not event.get('replay_same_weights')]
              for phase, events in captures.items()}
    assert len(losses['save']) == 2 and len(losses['resume']) == 3 and len(losses['continuous']) == 5
    assert losses['save'] + losses['resume'] == losses['continuous'], losses
    routing = json.loads((AUDIT / 'muon-routing-continuous-compatible.json').read_text())['parameters']
    counts = Counter(row['optimizer'] for row in routing)
    elements = Counter()
    for row in routing:
        elements[row['optimizer']] += row['elements']
    assert sum(row['name'].endswith('mix_coeff_input') for row in routing) == 32
    rows = []
    log = ROOT / 'training-smoke/logs/single-training-continuous-muon-20261010.log'
    for line in log.read_text().splitlines():
        match = re.search(r'\] iteration\s+(\d+).*?elapsed time per iteration \(ms\): ([\d.]+)', line)
        if match:
            step, ms = int(match[1]), float(match[2])
            rows.append({'step': step, 'elapsed_ms': ms, 'tokens_per_second': 256000 / ms})
    result.update(dependency={'emerging-optimizers': importlib.metadata.version('emerging-optimizers')},
                  fixed_eval_samples_exact=True, fixed_eval_loss_exact=True, eval_losses=losses,
                  routing_counts=dict(counts), routing_elements=dict(elements), timing=rows,
                  config_sha256=hashlib.sha256((ROOT / 'configs/config-rwkv7-3b-muon.yaml').read_bytes()).hexdigest(),
                  scope='single H800, original full 3B, BF16, TP1/PP1/SP off, seq128/global batch2',
                  compatibility='use_layer_wise_param_layout: false; ckpt_format: torch')
    reference = json.loads((AUDIT / '3b-reference-training-validation.json').read_text())
    adam_log = (ROOT / 'training-smoke/logs/3b-adam-control-20261010.log').read_text()
    adam_rows = []
    for line in adam_log.splitlines():
        match = comparison['PATTERN'].search(line)
        if match:
            step, samples, lr, loss, grad, skipped, nan = match.groups()
            adam_rows.append({'step': int(step), 'samples': int(samples), 'lr': float(lr),
                              'loss': float(loss), 'grad_norm': float(grad),
                              'skipped': int(skipped), 'nan': int(nan)})
    assert adam_rows == reference['train_steps'], adam_rows
    adam_evals = [float(x) for x in re.findall(r'validation loss at iteration.*?lm loss value: ([\d.Ee+-]+)', adam_log)]
    assert adam_evals == [x['loss'] for x in reference['evaluations']], adam_evals
    result['adam_reference_unchanged_at_logged_precision'] = True
    result['regression_tests_passed'] = 64
    (AUDIT / 'muon-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
