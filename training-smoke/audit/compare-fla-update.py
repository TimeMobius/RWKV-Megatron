"""Compare the frozen 3B single-GPU captures before/after the FLA update."""
import json
import argparse
import re
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parent


def compare(a, b, label=''):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, label
        diff = a.double() - b.double()
        assert torch.isfinite(b).all(), label
        return dict(exact=torch.equal(a, b), max_abs=diff.abs().max().item(),
                    relative_l2=(diff.norm() / a.double().norm().clamp_min(1e-30)).item())
    if isinstance(a, dict):
        assert a.keys() == b.keys(), label
        return {key: compare(value, b[key], label + '/' + key) for key, value in a.items()}
    return dict(exact=a == b, before=a, after=b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures-only', action='store_true')
    parser.add_argument('--report-only', action='store_true', help='Report known cross-version differences without requiring bit-exact compatibility')
    args = parser.parse_args()
    before = torch.load(PATH / 'full-before-fla-40850297.pth', map_location='cpu', weights_only=True)
    after = torch.load(PATH / 'full-after-fla-40850297.pth', map_location='cpu', weights_only=True)
    result = compare(before, after)
    def leaves(data):
        if 'exact' in data:
            yield data
        else:
            for value in data.values():
                yield from leaves(value)
    rows = list(leaves(result))
    report = dict(before_commit='79d12e8ec0ae72bc7f2189b4244d6532ce8c268c',
                  after_commit='40850297f9f977d7d33eb9d87fcf9a460554b553',
                  compared_leaves=len(rows), exact_leaves=sum(row['exact'] for row in rows),
                  all_exact=all(row['exact'] for row in rows), full_model_3b=True,
                  sequence=128, batch=1, dtype='bfloat16', gradient_scope='selected', results=result)
    (PATH / 'fla-update-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Exact comparisons', report['exact_leaves'], '/', report['compared_leaves'])
    print('loss', result['loss'], 'logits', result['logits'], 'v0', result['v0_gradient'])
    if not args.report_only:
        assert report['all_exact'], 'Investigate numerical changes before accepting the update.'
    if args.captures_only:
        return
    pattern = re.compile(r'\] iteration\s+(\d+)/\s*\d+.*?consumed samples:\s*(\d+)'
        r'.*?learning rate:\s*([\d.Ee+-]+).*?lm loss:\s*([\d.Ee+-]+).*?grad norm:\s*([\d.Ee+-]+)'
        r'.*?skipped iterations:\s*(\d+).*?nan iterations:\s*(\d+)')
    def train_rows(path):
        return [match.groups() for line in path.read_text().splitlines() if (match := pattern.search(line))]
    baseline = train_rows(PATH.parent / 'logs/single-training-continuous-te.log')
    candidate = train_rows(PATH.parent / 'logs/single-training-fla-40850297.log')
    assert len(baseline) == len(candidate) == 4
    training_exact = baseline == candidate
    if not args.report_only:
        assert training_exact, (baseline, candidate)
    assert all(row[-2:] == ('0', '0') for row in candidate), candidate
    assert all(left[:3] == right[:3] for left, right in zip(baseline, candidate)), (baseline, candidate)
    report['four_step_training_exact_at_logged_precision'] = training_exact
    report['baseline_training_rows'] = baseline
    report['training_rows'] = candidate
    (PATH / 'fla-update-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Four-step training loss/grad norm bit-exact:', training_exact, '; LR/sample progress match, zero NaN/skips.')


if __name__ == '__main__':
    main()
