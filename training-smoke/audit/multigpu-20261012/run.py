"""Sequential GPU audit jobs; independent CPU analysis may be run alongside."""
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
LAUNCH=ROOT/'training-smoke/audit/launch-final-single.sh'
LOG=ROOT/'training-smoke/logs'
status=[]
LEDGER="jobs.json"


def job(name,n,script,args,env=None):
    log=LOG/f'multigpu-{name}-20261012.log'
    cmd=['bash',str(LAUNCH),'-m','torch.distributed.run','--master_port','51313','--nproc_per_node',str(n),str(script),*args]
    if script.name=='distributed.py' and '--cases' in args:
        index=args.index('--cases');cases=args[index+1:]
        if len(cases)>1:
            for case in cases:job(name+'-'+case,n,script,args[:index]+['--cases',case],env)
            return
    print('START',name,flush=True)
    with log.open('w') as stream:
        result=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ| (env or {}))
    status.append(dict(name=name,gpus=n,exit_code=result.returncode,log=str(log)))
    (OUT/LEDGER).write_text(json.dumps(status,indent=2)+'\n')
    print('DONE',name,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':
    script=OUT/'distributed.py'
    job('full-fp32-dense',1,script,['--full-3b','--minipile','--sequence','32','--cases','dense-fp32-naive'])
    job('full-fp32-parallel',4,script,['--full-3b','--minipile','--sequence','32','--cases','tp2-fp32-naive','tp2-sp-fp32-naive','pp2-fp32-naive','tp2-pp2-fp32-naive','tp2-sp-pp2-fp32-naive','tp2-sp-pp2-recompute-fp32-naive'])
    job('full-bf16-dense',1,script,['--full-3b','--minipile','--cases','dense'])
    job('full-bf16-parallel',4,script,['--full-3b','--minipile','--cases','tp2','tp2-sp','pp2','tp2-pp2','tp2-sp-pp2','tp2-sp-pp2-recompute','tp2-sp-pp2-uniform2','tp2-sp-pp2-block2'])
    job('full-bf16-tp4',8,script,['--full-3b','--minipile','--cases','tp4-sp-pp2','tp4-sp-pp2-recompute'])
    for name,n in [('dense',1),('tp2',2),('tp2-sp',2),('pp2',2),('tp2-pp2',4)]:
        tag=name+'-nosave'
        job('train-'+tag,n,OUT/'training.py',['--rwkv-config',str(OUT/f'config-{tag}.yaml')],{'MULTIGPU_TAG':tag})
    for name in ['tp2-sp-pp2','tp4-sp-pp2']:
        for phase in ['save','resume','continuous']:
            tag=f'{name}-{phase}'
            job('train-'+tag,8,OUT/'training.py',['--rwkv-config',str(OUT/f'config-{tag}.yaml')],{'MULTIGPU_TAG':tag})
