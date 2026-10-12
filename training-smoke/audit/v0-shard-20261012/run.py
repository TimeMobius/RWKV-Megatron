"""Sequential distributed transport precision, continuation, and throughput trials."""
import importlib.util
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runner',OUT.parent/'multigpu-20261012/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.OUT=OUT;runner.LEDGER='jobs.json';runner.status=[]
job=runner.job


if __name__=='__main__':
    for mode in ('before','after'):
        for case,n in [('tp2-sp',2),('tp2-sp-pp2',4),('tp2-sp-pp2-recompute',4),('tp4-sp-pp2',8),('tp4-sp-pp2-recompute',8),('tp2-pp2',4),('tp2-sp-pp2-fp32-naive',4),('tp4-sp-pp2-fp32-naive',8)]:
            tag=f'small-{mode}-{case}'
            if mode=='before' and (OUT/tag/f'{case}.pth').exists():continue  # Reuse a completed frozen baseline.
            job('v0-'+tag,n,OUT/'distributed.py',['--cases',case],{'V0_CASE_TAG':tag,'V0_PAYLOAD_BEFORE':'1' if mode=='before' else '0'})
    for mode in ('before','after'):
        for fusion in ('0','1'):
            for case,n in [('tp2-sp-pp2-recompute',4),('tp4-sp-pp2-recompute',8)]:
                tag=f'full-{mode}-fusion{fusion}-{case}'
                if mode=='before' and (OUT/tag/f'full-3b-minipile-{case}.pth').exists():continue
                job('v0-'+tag,n,OUT/'distributed.py',['--full-3b','--minipile','--cases',case],{'V0_CASE_TAG':tag,'V0_PAYLOAD_BEFORE':'1' if mode=='before' else '0','V0_KEY_FUSION':fusion})
    for name in ('tp2-sp-pp2','tp4-sp-pp2'):
        for phase in ('save','resume','continuous'):
            tag=f'{name}-{phase}'
            job('v0-train-'+tag,8,OUT/'training.py',['--rwkv-config',str(OUT/f'config-{tag}.yaml')],{'MULTIGPU_TAG':tag})
    for mode in ('before','after'):
        for name in ('tp2-sp-pp2','tp4-sp-pp2'):
            tag=f'perf-{mode}-{name}'
            if mode=='before':continue  # Paired baseline already measured with identical settings.
            job('v0-'+tag,8,OUT/'training.py',['--rwkv-config',str(OUT/f'config-perf-{name}.yaml')],{'MULTIGPU_TAG':tag,'V0_PAYLOAD_BEFORE':'1' if mode=='before' else '0'})
