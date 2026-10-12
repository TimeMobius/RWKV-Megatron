import importlib.util
from pathlib import Path

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_audit_runner',OUT/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.status=[]
runner.LEDGER='extra-jobs.json'
def job(*a,**kw):
    try:runner.job(*a,**kw)
    except SystemExit as error:
        # Retain failing optional-mode evidence and continue independent diagnostics.
        print('OPTIONAL CASE FAILED',a[0],error,flush=True)


for case in ['tp2-sp-pp2-fp32-naive','tp2-sp-pp2-recompute-fp32-naive','tp2-sp-pp2-block2-fp32-naive','tp4-sp-pp2-fp32-naive']:
    job('small-extra-'+case,8,OUT/'distributed.py',['--cases',case])
for variant,n in [('shift',2),('shift',4),('singlemicro',2),('core_attn',4),('mlp',4),('layernorm',4),('saved',4)]:
    job(f'extra-{variant}-n{n}',n,OUT/'extra.py',['--variant',variant])
for variant in ('none','layernorm','mlp'):
    job('selective-fp32-'+variant,4,OUT/'extra.py',['--variant',variant,'--fp32','--microbatches','2'])
job('selective-layernorm-fixed',4,OUT/'extra.py',['--variant','layernorm','--microbatches','2'])
for key_mode,args in [('compiled',[]),('eager',['--eager-key'])]:
    job('recompute-replay-'+key_mode,1,OUT/'recompute-replay.py',args)
job('row-fp32-output',2,OUT/'row-precision.py',[])
for tag,n,pp,env in [('dense-gpu0',1,1,{}),('pp2',2,2,{}),('dense-gpu1',1,1,{'CUDA_VISIBLE_DEVICES':'1'})]:
    job('pp-numerics-'+tag,n,OUT/'pp-numerics.py',['--pp',str(pp),'--tag',tag],env)
for name,n in [('dense',1),('dp8',8),('tp2-sp-pp2',8),('tp4-sp-pp2',8)]:
    tag='perf-'+name
    job(tag,n,OUT/'training.py',['--rwkv-config',str(OUT/f'config-{tag}.yaml')],{'MULTIGPU_TAG':tag})
