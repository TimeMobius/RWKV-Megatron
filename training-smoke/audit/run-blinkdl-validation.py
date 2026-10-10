"""Serial single-GPU comparisons; preserve each exit and unmodified log."""
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'training-smoke/audit'


def main():
    diagnostic=['--bf16','--all-gradients','--reference-profile','diagnostic',
                '--reference-core','fla','--aligned-reference','--shared-pointops',
                '--te-reference-norms','--te-final-reference']
    cases={
        'full-fp32':['--full','--all-gradients','--naive-core','--reference-profile','blinkdl',
                     '--result-tag=-official-fp32'],
        'full-bf16':['--full','--all-gradients','--bf16','--reference-profile','blinkdl',
                     '--result-tag=-official-bf16'],
        'full-direct-vfirst':['--full',*diagnostic,'--result-tag=-official-direct-vfirst'],
        'small-direct-vfirst':[*diagnostic,'--result-tag=-official-direct-vfirst'],
        'full-bf16-torch-norms':['--full','--all-gradients','--bf16','--reference-profile','blinkdl',
                               '--torch-norms','--result-tag=-official-bf16-torch-norms'],
        'kernel':[],
    }
    results={}
    for name,args in cases.items():
        log=ROOT/f'training-smoke/logs/audit-blinkdl-{name}.log'
        launcher='launch-blinkdl-kernel.sh' if name=='kernel' else 'launch-single.sh'
        command=['bash',str(AUDIT/launcher),*args]
        with log.open('w') as output:
            result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        results[name]={'command':command,'log':str(log),'exit_code':result.returncode}
        (AUDIT/'blinkdl-validation-exits.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,'exit',result.returncode,flush=True)
        # Kernel numerical failures should remain reportable rather than stop
        # preservation of independent full-model comparisons.
        if result.returncode and name!='kernel':
            raise SystemExit(result.returncode)


if __name__=='__main__':main()
