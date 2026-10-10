"""Real 3B optimizer/checkpoint continuation with L2Wrap and two microbatches."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    results = {}
    for phase in ('save', 'resume', 'continuous'):
        config = ROOT/f'training-smoke/config-3b-single-{phase}-l2wrap.yaml'
        command = ['bash', str(ROOT/'training-smoke/audit/launch-training-l2wrap.sh'),
                   f'single-training-{phase}-l2wrap', str(config)]
        checkpoint = ROOT/'training-smoke/checkpoints'/(
            'single-continuous-l2wrap' if phase == 'continuous' else 'single-resume-l2wrap')
        if phase != 'resume' and checkpoint.exists():
            raise RuntimeError(f'Use a new checkpoint location for this trial: {checkpoint}')
        log = ROOT/f'training-smoke/logs/l2wrap-runner-{phase}.log'
        with log.open('w') as output:
            completed = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        results[phase] = {'command': command, 'exit_code': completed.returncode,
                          'l2wrap_coefficient': 1e-4, 'key_fusion_config': True, 'microbatches': 2}
        (ROOT/'training-smoke/audit/l2wrap-training-exits.json').write_text(json.dumps(results, indent=2)+'\n')
        print(phase, completed.returncode, flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
