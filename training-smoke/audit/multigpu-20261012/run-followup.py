import importlib.util
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_audit_runner',OUT/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.status=[];runner.LEDGER='followup-jobs.json'
for variant in ('none','layernorm','mlp'):
    runner.job('selective-fp32-'+variant,4,OUT/'extra.py',['--variant',variant,'--fp32','--microbatches','2'])
runner.job('selective-layernorm-fixed',4,OUT/'extra.py',['--variant','layernorm','--microbatches','2'])
runner.job('selective-layernorm-training',8,OUT/'training.py',['--rwkv-config',str(OUT/'config-layernorm-recompute-fixed.yaml')],{'MULTIGPU_TAG':'layernorm-fixed'})
runner.job('recompute-replay-parallel',4,OUT/'recompute-replay.py',['--tp','2','--pp','2','--sp'])

runner.job('row-fp32-output-with-backward',2,OUT/'row-precision.py',[])

runner.job('pp2-bf16-repeat-with-backward',2,OUT/'distributed.py',['--full-3b','--minipile','--cases','pp2'])
