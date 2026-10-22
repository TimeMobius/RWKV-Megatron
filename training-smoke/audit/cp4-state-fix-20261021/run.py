"""Sequential real-model checks after the operator gate, with an execution ledger."""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
cases = [
    ('strict-cp4-sp-16k',{'CP_VALIDATION_WKV':'1'}),
    ('tp2-pp2-cp2-sp-mb2',{'CP_VALIDATION_WKV':'1'}),
    ('capture-cp4-8k',{'CP_FIX_ENTRY':'training-smoke/audit/cp-equivalence-20261021/capture.py',
                        'CP_AUDIT_OUTPUT':str(HERE/'captures'/'cp4-8k'),'CP_VALIDATION_WKV':'0'}),
    ('timing-affine',{'CP_FIX_AFFINE_BASELINE':'1','CP_VALIDATION_WKV':'0'}),
    ('timing-recurrent',{'CP_FIX_AFFINE_BASELINE':'0','CP_VALIDATION_WKV':'0'}),
]
ledger = []
for name, env in cases:
    print('START',name,flush=True)
    start = time.time()
    result = subprocess.run(['bash',str(HERE/'launch.sh'),name,'8'],cwd=ROOT,env=os.environ|env)
    ledger.append(dict(name=name,exit_code=result.returncode,seconds=time.time()-start,env=env))
    (HERE/'jobs.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print('DONE',name,result.returncode,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
