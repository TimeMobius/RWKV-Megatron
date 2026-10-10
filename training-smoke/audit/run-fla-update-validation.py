"""Serial GPU checks for the fast-forward from FLA79d12e8e to40850297."""
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'training-smoke'
config = yaml.safe_load((PATH / 'config-3b-single-continuous-te.yaml').read_text())
config.update(save=None, load=None, save_interval=1000000, exit_interval=None)
config_path = PATH / 'config-3b-single-fla-40850297.yaml'
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
tasks = [
    ('upstream-rwkv7-fla-40850297', ['bash', 'training-smoke/audit/launch-fla-update-tests.sh']),
    ('single-training-fla-40850297', ['bash', 'training-smoke/audit/launch-training-single.sh',
                                   'single-training-fla-40850297', str(config_path)]),
    ('compare-fla-40850297', ['/mnt/data/anaconda3/envs/train_mobius_py312/bin/python',
                           'training-smoke/audit/compare-fla-update.py', '--report-only']),
]
results = {}
for name, command in tasks:
    print('START', name, flush=True)
    with (PATH / 'logs' / f'audit-{name}.log').open('w') as console:
        proc = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
    results[name] = dict(command=command, exit_code=proc.returncode)
    (PATH / 'audit/fla-update-exits.json').write_text(json.dumps(results, indent=2) + '\n')
    print('END', name, proc.returncode, flush=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)
