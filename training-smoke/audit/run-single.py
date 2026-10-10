"""Run only single-GPU reference and checkpoint validations sequentially."""
import json
import subprocess
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
TASKS=[
 ('single-small-fp32-fla',['bash','training-smoke/audit/launch-single.sh']),
 ('single-small-bf16-fla',['bash','training-smoke/audit/launch-single.sh','--bf16','--reference-core','fla']),
 ('single-small-bf16-recurrent',['bash','training-smoke/audit/launch-single.sh','--bf16']),
 ('single-full-fp32-naive',['bash','training-smoke/audit/launch-single.sh','--full','--naive-core']),
 ('single-full-bf16-fla',['bash','training-smoke/audit/launch-single.sh','--full','--bf16','--reference-core','fla']),
 *[(f'single-training-{phase}',['bash','training-smoke/audit/launch-training-single.sh',f'single-training-{phase}',f'training-smoke/config-3b-single-{phase}.yaml'])
   for phase in ['save','resume','continuous']],
]
statuses={}
for name,cmd in TASKS:
 print('START',name,flush=True)
 start=datetime.now(timezone.utc).isoformat()
 with (ROOT/f'training-smoke/logs/audit-{name}.log').open('w') as console:
  result=subprocess.run(cmd,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT)
 statuses[name]={'command':cmd,'exit_code':result.returncode,'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat()}
 (ROOT/'training-smoke/audit/single-exits.json').write_text(json.dumps(statuses,indent=2)+'\n')
 print('DONE',name,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
