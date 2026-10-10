"""Serial full-model candidate tests without changing training defaults."""
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'training-smoke/audit'


def main():
    cases={'model-benchmark':['launch-fla-model-benchmark.sh']}
    for mode in ('correction','norm','norm-correction','wkv-save'):
        cases[mode]=['launch-single.sh','--full','--bf16','--all-gradients','--reference-profile','blinkdl',
                     '--fla-candidate',mode,'--capture-output','full-fla-candidate-'+mode,
                     '--result-tag=-fla-candidate-evaluation']
    results={}
    for name,args in cases.items():
        command=['bash',str(PATH/args[0]),*args[1:]]
        log=ROOT/f'training-smoke/logs/audit-fla-candidate-{name}.log'
        with log.open('w') as output:
            process=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        results[name]={'command':command,'exit_code':process.returncode,'log':str(log)}
        (PATH/'fla-candidate-exits.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,'exit',process.returncode,flush=True)
        if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':main()
