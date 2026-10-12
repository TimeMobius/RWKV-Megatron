import importlib.util
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_audit_runner',OUT/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.status=json.loads((OUT/'jobs.json').read_text())
runner.status[-1]['log']=str(runner.LOG/'multigpu-train-instrumentation-import-failure-20261012.log')
for name in ['tp2-sp-pp2','tp4-sp-pp2']:
    for phase in ['save','resume','continuous']:
        tag=f'{name}-{phase}'
        runner.job('train-'+tag,8,OUT/'training.py',['--rwkv-config',str(OUT/f'config-{tag}.yaml')],{'MULTIGPU_TAG':tag})
