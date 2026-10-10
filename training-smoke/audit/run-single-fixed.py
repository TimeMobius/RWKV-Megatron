"""Validate corrected point-op layout on single-GPU native/reference/resume runs."""
import json
import subprocess
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'training-smoke'
tasks=[
 ('single-fixed-small-fp32',['bash','training-smoke/audit/launch-single.sh','--naive-core']),
 ('single-fixed-full-bf16',['bash','training-smoke/audit/launch-single.sh','--full','--bf16','--reference-core','fla','--aligned-reference','--apex-reference-norms']),
]
for phase in ('save','resume','continuous'):
 config=yaml.safe_load((P/f'config-3b-single-{phase}.yaml').read_text())
 for key in ('save','load'):
  if config.get(key):config[key]+='-fixed'
 path=P/f'config-3b-single-{phase}-fixed.yaml'
 path.write_text(yaml.safe_dump(config,sort_keys=False))
 tasks.append((f'single-training-{phase}-fixed',['bash','training-smoke/audit/launch-training-single.sh',f'single-training-{phase}-fixed',str(path)]))
statuses={}
for name,command in tasks:
 print('START',name,flush=True)
 with (P/f'logs/audit-{name}.log').open('w') as console:
  result=subprocess.run(command,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT)
 statuses[name]={'command':command,'exit_code':result.returncode}
 (P/'audit/single-fixed-exits.json').write_text(json.dumps(statuses,indent=2)+'\n')
 print('END',name,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
