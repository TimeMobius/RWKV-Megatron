"""Summarize paired six-step training trials without counting compile warmup."""
import json
import re
import statistics
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def trial(mode,name):
    path=ROOT/f'training-smoke/logs/multigpu-v0-perf-{mode}-{name}-20261012.log'
    values=[(int(step),float(ms)) for step,ms in re.findall(r'\] iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+)',path.read_text())]
    assert len(values)==6,(path,values)
    warm=[ms for step,ms in values if step>=2]
    med=statistics.median(warm)
    peaks=[]
    for rank in range(8):
        records=json.loads((OUT/f'perf-{mode}-{name}-rank{rank}-events.json').read_text())
        assert len([r for r in records if r['kind']=='train'])==6
        assert all(r['skipped']==0 for r in records if r['kind']=='train')
        evaluation=[r for r in records if r['kind']=='eval'][-1]
        peaks.append(evaluation['cuda_peak_mib'])
    return {'median_ms':med,'warm_ms':warm,'total_tps':16384*1000/med,'peak_allocated_mib_by_rank':peaks}


def main():
    results={}
    for name in ('tp2-sp-pp2','tp4-sp-pp2'):
        before,after=trial('before',name),trial('after',name)
        results[name]={'before':before,'after':after,'tps_gain_percent':(after['total_tps']/before['total_tps']-1)*100,
                       'peak_allocated_max_mib_saved':max(before['peak_allocated_mib_by_rank'])-max(after['peak_allocated_mib_by_rank']),
                       'scope':'8 H800, seq2048, microbatch1/globalbatch8, six steps, median2-6, BF16/Adam/uniform1/default key fusion'}
    (OUT/'performance.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
