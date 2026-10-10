"""Serial full-model probes; diagnostic changes are process-local only."""
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'training-smoke/audit'


def main():
    diagnostic=['--full','--bf16','--all-gradients','--reference-profile','diagnostic',
                '--reference-core','fla','--aligned-reference','--shared-pointops',
                '--te-reference-norms','--te-final-reference','--audit-v0-edges']
    official=['--full','--bf16','--all-gradients','--reference-profile','blinkdl']
    cases={
        'v0-baseline':[*diagnostic,'--capture-output=full-v0-bf16-baseline','--result-tag=-precision-v0-baseline'],
        'v0-fp32':[*diagnostic,'--v0-fp32','--capture-output=full-v0-fp32-prototype','--result-tag=-precision-v0-prototype'],
        'key':[*official,'--precision-probe','key','--result-tag=-precision-intermediates'],
        'all':[*official,'--precision-probe','all','--result-tag=-precision-intermediates'],
        'all-v0':[*official,'--precision-probe','all','--v0-fp32','--result-tag=-precision-intermediates'],
        'fp32-oracle':['--full','--all-gradients','--naive-core','--reference-profile','blinkdl',
                      '--capture-output=full-fp32-current-precision-reference',
                      '--capture-reference-output=full-fp32-independent-precision-reference',
                      '--result-tag=-precision-oracle'],
    }
    results={}
    for name,args in cases.items():
        log=ROOT/f'training-smoke/logs/audit-precision-full-{name}.log'
        command=['bash',str(AUDIT/'launch-single.sh'),*args]
        with log.open('w') as output:
            process=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        results[name]={'command':command,'log':str(log),'exit_code':process.returncode}
        (AUDIT/'precision-probe-exits.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,'exit',process.returncode,flush=True)
        if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':main()
