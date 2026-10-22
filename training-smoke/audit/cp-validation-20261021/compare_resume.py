"""Assert complete parameter/Adam shard hashes and exact continuation events."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def compare(sp, reference=None, actual=None, output=None):
    a = reference or HERE/f'resume-sp{sp}-continuous'
    b = actual or HERE/f'resume-sp{sp}-restored'
    rows = []
    for rank in range(8):
        states = {}
        for label in ('before3','after4'):
            left, right = [json.loads((p/f'{label}-rank{rank}.json').read_text()) for p in (a,b)]
            assert left.keys() == right.keys()
            mismatches = [k for k in left if left[k] != right[k]]
            states[label] = dict(records=len(left), mismatches=mismatches)
            assert not mismatches, (sp, rank, label, mismatches[:10])
        left, right = [json.loads((p/f'rank{rank}.json').read_text()) for p in (a,b)]
        expected = [{k:v for k,v in e.items() if k != 'peak_allocated_gib'} for e in left['events'] if e['iteration'] >= 3]
        actual = [{k:v for k,v in e.items() if k != 'peak_allocated_gib'}
                  for e in right['events'] if e['iteration'] >= 3]
        assert expected == actual, (sp, rank, 'continuation events differ')
        rows.append(dict(rank=rank, states=states, continuation_events_exact=True,
                         events_compared=len(actual)))
    result = dict(sp=bool(sp), checkpoint_step=2, resumed_steps=[3,4], passed=True, ranks=rows)
    (output or HERE/f'resume-sp{sp}-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'SP{sp}: all 8 ranks, parameters/FP32 main weights/Adam tensor states/scheduler and events exact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sp', type=int, choices=(0,1))
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--actual', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    compare(args.sp, args.reference, args.actual, args.output)
