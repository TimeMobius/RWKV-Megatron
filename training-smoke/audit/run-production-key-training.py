"""Real 3B optimizer/checkpoint continuation for production key fusion."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    results = {}
    for phase in ('save', 'resume', 'continuous'):
        config = ROOT/f'training-smoke/config-3b-single-{phase}-key-fusion.yaml'
        command = ['bash', str(ROOT/'training-smoke/audit/launch-training-key-fusion.sh'),
                   f'single-training-{phase}-key-fusion', str(config)]
        checkpoint = ROOT/'training-smoke/checkpoints'/(
            'single-continuous-key-fusion' if phase == 'continuous' else 'single-resume-key-fusion')
        if phase != 'resume' and checkpoint.exists():
            raise RuntimeError(f'Use a new checkpoint location for this trial: {checkpoint}')
        log = ROOT/f'training-smoke/logs/key-production-runner-{phase}.log'
        with log.open('w') as output:
            completed = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        results[phase] = {'command': command, 'exit_code': completed.returncode,
                          'fusion_enabled': os.environ.get('RWKV7_KEY_FUSION', '1') != '0'}
        (ROOT/'training-smoke/audit/production-key-training-exits.json').write_text(json.dumps(results, indent=2)+'\n')
        print(phase, completed.returncode, flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
