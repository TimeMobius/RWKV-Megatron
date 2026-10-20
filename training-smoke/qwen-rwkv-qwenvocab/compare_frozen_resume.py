"""Compare full-precision traces and complete final training states."""

import io
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).parent / 'frozen-resume-20261020'


def compare_state(a, b):
    result = dict(tensors=0, elements=0, mismatches=[])

    def visit(x, y, path):
        if type(x) is not type(y):
            result['mismatches'].append(path + ':type')
        elif isinstance(x, torch.Tensor):
            result['tensors'] += 1
            result['elements'] += x.numel()
            if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
                item = dict(path=path, shape=list(x.shape), dtype=str(x.dtype))
                if x.shape == y.shape and x.numel():
                    item['max_abs'] = float((x.float() - y.float()).abs().max())
                result['mismatches'].append(item)
        elif isinstance(x, dict):
            if x.keys() != y.keys():
                result['mismatches'].append(path + ':keys')
            for key in x.keys() & y.keys():
                visit(x[key], y[key], path + '.' + str(key))
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y):
                result['mismatches'].append(path + ':length')
            for i, (xx, yy) in enumerate(zip(x, y)):
                visit(xx, yy, path + '.' + str(i))
        elif isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                result['mismatches'].append(path)
        elif isinstance(x, io.BytesIO):
            if x.getvalue() != y.getvalue():
                result['mismatches'].append(path)
        elif x != y:
            result['mismatches'].append(path)

    visit(a, b, 'state')
    result['equal'] = not result['mismatches']
    return result


def main():
    torch.set_num_threads(8)
    a = json.loads((FOLDER / 'record-trace.json').read_text())
    b = json.loads((FOLDER / 'replay-trace.json').read_text())
    aa = {x['step']: x for x in a['steps']}
    bb = {x['step']: x for x in b['steps']}
    assert set(aa) == set(range(501, 506))
    assert set(bb) == set(range(502, 506))
    rows = []
    for step in sorted(bb):
        x, y = aa[step], bb[step]
        assert x['teacher_key'] == y['teacher_key'] and y['teacher_source'] == 'disk'
        logits_a = torch.load(x['logits_path'], map_location='cpu', weights_only=True)
        logits_b = torch.load(y['logits_path'], map_location='cpu', weights_only=True)
        rows.append(dict(step=step, continuous_kl=x['kl'], resumed_kl=y['kl'],
                         kl_abs_diff=abs(x['kl'] - y['kl']),
                         continuous_grad_norm=x['grad_norm'], resumed_grad_norm=y['grad_norm'],
                         grad_norm_abs_diff=abs(x['grad_norm'] - y['grad_norm']),
                         logits_equal=torch.equal(logits_a, logits_b),
                         logits_max_abs=float((logits_a.float() - logits_b.float()).abs().max()),
                         skipped_continuous=x['skipped'], skipped_resumed=y['skipped']))
    checkpoints = [ROOT / f'training-smoke/checkpoints/frozen-resume-20261020-{mode}/iter_0000505/mp_rank_00/model_optim_rng.pt'
                   for mode in ('record', 'replay')]
    states = [torch.load(p, map_location='cpu', mmap=True, weights_only=False) for p in checkpoints]
    sections = {}
    for key in ('iteration', 'model', 'optimizer', 'opt_param_scheduler', 'rng_state'):
        sections[key] = compare_state(states[0][key], states[1][key])
        print(key, sections[key], flush=True)
    report = dict(scope='2.74B single GPU BF16/FLA, all parameters trained. Record 501-505 from existing step500; save at501; restart at501 and replay502-505 with HTTP forbidden.',
                  teacher_files=len(list((FOLDER / 'teacher').glob('*.pt'))),
                  checkpoints=[str(p) for p in checkpoints], steps=rows, final_states=sections,
                  all_training_states_equal=all(sections[k]['equal'] for k in ('iteration', 'model', 'optimizer', 'opt_param_scheduler')),
                  all_losses_equal=all(x['kl_abs_diff'] == 0 for x in rows),
                  all_logits_equal=all(x['logits_equal'] for x in rows),
                  all_grad_norms_equal=all(x['grad_norm_abs_diff'] == 0 for x in rows))
    output = FOLDER.parent / 'frozen-resume-results-20261020.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
