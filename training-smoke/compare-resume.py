#!/usr/bin/env python3
"""Check that resumed training reproduces uninterrupted loss and gradient norms."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = re.compile(
    r'\] iteration\s+(\d+)/\s*\d+.*?consumed samples:\s*(\d+)'
    r'.*?learning rate:\s*([\d.Ee+-]+).*?lm loss:\s*([\d.Ee+-]+)'
    r'.*?grad norm:\s*([\d.Ee+-]+).*?skipped iterations:\s*(\d+)'
    r'.*?nan iterations:\s*(\d+)'
)


def rows(name):
    lines = (ROOT / f'logs/{name}.log').read_text(errors='replace').splitlines()
    result = {}
    for line in lines:
        match = TRAIN.search(line)
        if match:
            step, samples, lr, loss, grad, skipped, nan = match.groups()
            result[int(step)] = dict(samples=int(samples), lr=float(lr), loss=float(loss),
                                     grad_norm=float(grad), skipped=int(skipped), nan=int(nan))
    return result


def main():
    exits = json.loads((ROOT / 'validation-exits.json').read_text())
    for name in ('checkpoint-save', 'checkpoint-resume', 'checkpoint-continuous'):
        assert exits[name]['exit_code'] == 0, exits[name]
    saved, resumed, uninterrupted = [rows(name) for name in
                                    ('checkpoint-save', 'checkpoint-resume', 'checkpoint-continuous')]
    assert sorted(saved) == [1, 2], saved
    assert sorted(resumed) == [3, 4], resumed
    assert sorted(uninterrupted) == [1, 2, 3, 4], uninterrupted
    combined = saved | resumed
    for step, row in uninterrupted.items():
        assert combined[step] == row, (step, combined[step], row)
        assert row['skipped'] == row['nan'] == 0, row
    text = (ROOT / 'logs/checkpoint-resume.log').read_text(errors='replace')
    assert 'successfully loaded checkpoint' in text, 'Missing checkpoint-load confirmation'
    result = dict(exact_at_logged_precision=True, saved_steps=[1, 2], resumed_steps=[3, 4],
                  compared_metrics=['samples', 'lr', 'loss', 'grad_norm', 'skipped', 'nan'],
                  uninterrupted=uninterrupted)
    (ROOT / 'results-resume.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
