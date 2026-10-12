"""Independent FP32 and norm-selective coverage for the final local-head payload."""
import importlib.util
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runner',OUT.parent/'multigpu-20261012/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.OUT=OUT;runner.LEDGER='final-jobs.json';runner.status=[]
for mode in ('before','after'):
    tag=f'full-{mode}-fp32-tp4-sp-pp2'
    runner.job('v0-'+tag,8,OUT/'distributed.py',['--full-3b','--minipile','--sequence','32','--cases','tp4-sp-pp2-fp32-naive'],
               {'V0_CASE_TAG':tag,'V0_PAYLOAD_BEFORE':'1' if mode=='before' else '0'})
    tag=f'selective-{mode}-fp32-tp2-sp-pp2'
    runner.job('v0-'+tag,4,OUT/'distributed.py',['--cases','tp2-sp-pp2-fp32-naive'],
               {'V0_CASE_TAG':tag,'V0_PAYLOAD_BEFORE':'1' if mode=='before' else '0','V0_SELECTIVE_NORM':'1'})

for mode in ('before','after'):
    tag=f'pp4-{mode}-tp2-sp-pp4'
    runner.job('v0-'+tag,8,OUT/'extra.py',[],{'V0_CASE_TAG':tag,'V0_PAYLOAD_BEFORE':'1' if mode=='before' else '0'})
