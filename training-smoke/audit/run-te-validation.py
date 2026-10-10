"""Serial single-GPU validation of unified TE norms; separate evidence paths."""
import json
import argparse
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'training-smoke'
parser = argparse.ArgumentParser()
parser.add_argument('--from-task')
args = parser.parse_args()
tasks = [('te-full-fp32', ['bash', 'training-smoke/audit/launch-single.sh', '--full',
          '--naive-core', '--all-gradients', '--result-tag=-unified-te'])]
for mode in ('te', 'old', 'missing'):
    tasks.append((f'te-fallback-{mode}', ['bash', 'training-smoke/audit/launch-te-fallback.sh', '--mode', mode]))
for phase in ('save', 'resume', 'continuous'):
    config = yaml.safe_load((PATH / f'config-3b-single-{phase}-fixed.yaml').read_text())
    for key in ('save', 'load'):
        if config.get(key):
            config[key] = config[key].removesuffix('-fixed') + '-te'
    target = PATH / f'config-3b-single-{phase}-te.yaml'
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    tasks.append((f'single-training-{phase}-te', ['bash', 'training-smoke/audit/launch-training-single.sh',
                                              f'single-training-{phase}-te', str(target)]))
tasks.append(('te-resume-compare', ['/mnt/data/anaconda3/envs/train_mobius_py312/bin/python',
                                  'training-smoke/audit/compare-single-resume.py', '--suffix=-te']))
record = PATH / 'audit/te-validation-exits.json'
results = json.loads(record.read_text()) if args.from_task and record.exists() else {}
if args.from_task:
    tasks = tasks[next(i for i, (name, _) in enumerate(tasks) if name == args.from_task):]
for name, command in tasks:
    print('START', name, flush=True)
    with (PATH / 'logs' / f'audit-{name}.log').open('w') as console:
        result = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
    results[name] = dict(command=command, exit_code=result.returncode)
    (PATH / 'audit/te-validation-exits.json').write_text(json.dumps(results, indent=2) + '\n')
    print('END', name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
