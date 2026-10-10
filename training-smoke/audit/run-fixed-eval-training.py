"""Serial real3B fixed/rolling validation and fixed-validation resume trial."""
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    results={}
    for phase in ('save','resume','continuous','rolling'):
        checkpoint=ROOT/'training-smoke/checkpoints'/(
            'single-continuous-fixed-eval' if phase=='continuous' else 'single-resume-fixed-eval')
        if phase in ('save','continuous') and checkpoint.exists():
            raise RuntimeError(f'Choose a new output directory: {checkpoint}')
        config=ROOT/f'training-smoke/config-3b-{phase}-fixed-eval.yaml'
        command=['bash',str(ROOT/'training-smoke/audit/launch-training-fixed-eval.sh'),
                 f'single-training-{phase}-fixed-eval',str(config)]
        env=dict(os.environ,FIXED_EVAL_CAPTURE=f'fixed-eval-{phase}-batches.json')
        log=ROOT/f'training-smoke/logs/fixed-eval-runner-{phase}.log'
        with log.open('w') as output:
            proc=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        results[phase]={'command':command,'exit_code':proc.returncode,'log':str(log),
                       'capture':env['FIXED_EVAL_CAPTURE']}
        (ROOT/'training-smoke/audit/fixed-eval-training-exits.json').write_text(json.dumps(results,indent=2)+'\n')
        print(phase,proc.returncode,flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
