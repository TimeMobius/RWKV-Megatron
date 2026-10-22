"""Sequential GPU jobs with an exit-code ledger, bounded per-job runtime."""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
jobs = [
    ('tp2-pp1-cp2', 4, {'CP_VALIDATION_WKV': '1'}),
    ('tp2-pp1-cp4-8k', 8, {'CP_VALIDATION_WKV': '1'}),
    ('tp2-pp1-cp4-8k-measured', 8, {'CP_VALIDATION_WKV': '1', 'CP_VALIDATION_WKV_ENFORCE': '0'}),
    ('tp2-pp1-cp4-sp-16k', 8, {}),
    ('resume-sp0-continuous', 8, {'CP_VALIDATION_STATE': '1'}),
    ('resume-sp0-restored', 8, {'CP_VALIDATION_STATE': '1'}),
    ('resume-sp1-continuous', 8, {'CP_VALIDATION_STATE': '1'}),
    ('resume-sp1-restored', 8, {'CP_VALIDATION_STATE': '1'}),
    ('fp8', 8, {}),
    ('muon', 8, {}),
    ('fp8-muon', 8, {}),
    ('fp8-delayed', 8, {}),
    ('ordinary-regression-sp1', 8, {'CP_VALIDATION_STATE': '1'}),
    *[(f'capture-cp{cp}-8k', 2*cp, {
        'CP_VALIDATION_ENTRY': 'training-smoke/audit/cp-equivalence-20261021/capture.py',
        'CP_AUDIT_OUTPUT': str(HERE/'captures'/f'cp{cp}-8k')}) for cp in (1, 4)],
]
ledger_path = HERE/'jobs.json'
ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
for name, n, env in jobs:
    print('START', name, flush=True)
    start = time.time()
    try:
        result = subprocess.run(['bash', str(HERE/'launch.sh'), name, str(n)], cwd=ROOT,
                                env=os.environ | env, timeout=1800)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 'timeout'
        # Do not overlap later jobs with unresolved GPU workers.
        ledger.append(dict(name=name, exit_code=code))
        ledger_path.write_text(json.dumps(ledger, indent=2)+'\n')
        raise
    ledger.append(dict(name=name, exit_code=code, seconds=time.time()-start, env=env))
    ledger_path.write_text(json.dumps(ledger, indent=2)+'\n')
    print('DONE', name, code, flush=True)
