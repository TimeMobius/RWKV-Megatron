"""Verify strict WKV gates and summarize controlled whole-training timings."""
import json
import re
import statistics
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def strict_case(name,ranks):
    paths = sorted((HERE/name).glob('rank*.json'))
    assert len(paths)==ranks, (name,len(paths))
    records = [json.loads(p.read_text()) for p in paths]
    wkv = [row for record in records for row in record['wkv'].values()]
    config = yaml.safe_load((HERE/f'config-{name}.yaml').read_text())
    pp = config['model_parallel']['pipeline_model_parallel_size']
    assert len(wkv)==24*ranks//pp,(name,len(wkv))
    assert {layer for record in records for layer in record['wkv']} == {str(i) for i in range(24)}
    assert all(row['passed'] and row['output']['exact'] for row in wkv)
    assert all(g['exact'] for row in wkv for g in row['gradients'].values())
    steps = [e for record in records for e in record['events'] if e['kind']=='train']
    assert len(steps)==3*ranks and all(e['skipped']==0 for e in steps)
    return dict(ranks=ranks,layers=24,total_heads=36,layer_rank_records=len(wkv),
                outputs_and_all_six_input_gradients_exact=True,optimizer_steps=3,skipped_steps=0)


def main():
    result = dict(strict={n:strict_case(n,r) for n,r in (
        ('strict-cp2-4k',4),('strict-cp4-8k',8),('strict-cp4-sp-16k',8),('tp2-pp2-cp2-sp-mb2',8))})
    result['timing'] = {}
    assert (HERE/'config-timing-affine.yaml').read_bytes()==(HERE/'config-timing-recurrent.yaml').read_bytes()
    for mode in ('affine','recurrent'):
        text = (HERE/f'timing-{mode}/run.log').read_text()
        values = [(int(step),float(ms)) for step,ms in re.findall(
            r'iteration\s+(\d+)\s*/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)',text)]
        steady = [ms for step,ms in values if step>=3]
        assert len(steady)==6,(mode,values)
        result['timing'][mode] = dict(iterations=values,median_steps_3_to_8_ms=statistics.median(steady))
    for rank in range(8):
        traces=[json.loads((HERE/f'timing-{mode}'/f'rank{rank}.json').read_text())
                for mode in ('affine','recurrent')]
        assert all(len(trace['events'])==8 for trace in traces)
        assert all(e['skipped']==0 for trace in traces for e in trace['events'])
        assert [e['batches'] for e in traces[0]['events']]==[e['batches'] for e in traces[1]['events']]
    result['timing']['configs_and_all_rank_batches_equal']=True
    result['timing']['recurrent_over_affine'] = result['timing']['recurrent']['median_steps_3_to_8_ms']/result['timing']['affine']['median_steps_3_to_8_ms']
    model = json.loads((HERE/'comparison-model.json').read_text())
    assert model['passed'] and all(row['max_abs']==0 for row in model['forward'].values())
    assert model['gradient_aggregate']['relative_l2']<0.0005
    result['model'] = dict(forward_capture_exact=True,gradient_aggregate=model['gradient_aggregate'])
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
