import importlib.util
from pathlib import Path
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runner',OUT/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.status=[];runner.LEDGER='final-extra-jobs.json'
runner.job('recompute-replay-parallel-eager',4,OUT/'recompute-replay.py',['--tp','2','--pp','2','--sp','--eager-key'])
runner.job('bottleneck-distributed-regression',2,OUT.parents[2]/'tests/test_rwkv7_bottleneck_tp_batchmix_distributed.py',[])

runner.job('pp2-bf16-repeat-with-backward',2,OUT/'distributed.py',['--full-3b','--minipile','--cases','pp2'])
