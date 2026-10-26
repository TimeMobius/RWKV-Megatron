"""Numerical alignment of pooled replacement through production GPU schedules."""

import json
import sys
from pathlib import Path

import torch
from compare_dp_distill import canonical, metric, samples

ROOT = Path(__file__).resolve().parents[1]


def precision(folder, tag, dp, tp, pp):
    def load(name):
        return [row for path in sorted((folder / name).glob('step1-rank*.pt'))
                if (row := torch.load(path, weights_only=False))['cp'] == 0]

    reference, actual = load('dense'), load(tag)
    ref_samples, ref_logits = samples(reference, 1, 1)
    act_samples, act_logits = samples(actual, dp, pp)
    assert ref_samples == act_samples
    keys = sorted(ref_samples)
    logits = metric(torch.stack([act_logits[k] for k in keys]),
                    torch.stack([ref_logits[k] for k in keys]))
    assert logits['finite'] and logits['relative_l2'] < 0.02, (tag, logits)
    expected = canonical(reference, 'grads', 1, 1)
    grads = canonical(actual, 'grads', tp, pp)
    metrics = {key: metric(grads[key], expected[key]) for key in expected}
    assert all(value['finite'] for value in metrics.values())
    significant = {key: value for key, value in metrics.items() if value['reference_l2'] >= 1e-3}
    worst = max(significant, key=lambda key: significant[key]['relative_l2'])
    assert significant[worst]['relative_l2'] < 0.05, (tag, worst, significant[worst])
    final = next(row for row in actual if row['dp'] == 0 and row['pp'] == pp - 1 and row['tp'] == 0)
    errors = {key: abs(final['losses'][key] - reference[0]['losses'][key]) / abs(reference[0]['losses'][key])
              for key in ('lm loss', 'distill/kl_loss', 'distill/student_ce_loss')}
    assert max(errors.values()) < 0.005, (tag, errors)
    return dict(logits_relative_l2=logits['relative_l2'],
                worst_significant_gradient=worst,
                worst_significant_gradient_relative_l2=significant[worst]['relative_l2'],
                loss_relative_errors=errors)


if __name__ == '__main__':
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'training-smoke/distill-remaining-20261026/teacher-pool-replacement'
    report = {tag: precision(folder, tag, dp, 2, 2)
              for tag, dp in [('combined', 2), ('cp-combined', 1)]}
    (folder / 'precision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
