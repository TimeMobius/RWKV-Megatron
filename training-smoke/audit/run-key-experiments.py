"""Serial single-GPU experiments; original inputs and production code untouched."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'training-smoke/audit'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--exits-file',default='key-experiment-exits.json')
    options=parser.parse_args()
    tokens=str(AUDIT/'key-minipile-4097.pth')
    cases={'operator':['launch-key-experiment.sh','operator'],
           'model-benchmark':['launch-key-experiment.sh','model']}
    for offset in (0,512):
        for mode in ('bf16','key','key-fused'):
            name=f'{mode}-seq512-offset{offset}'
            args=['launch-single.sh','--full','--bf16','--all-gradients','--reference-profile','blinkdl',
                  '--sequence-length','512','--token-offset',str(offset),'--tokens-file',tokens,
                  '--result-tag=-key-fusion-'+name]
            if mode!='bf16':args+=['--precision-probe',mode,'--capture-output','full-key-fusion-'+name,
                                  '--capture-reference-output','full-key-fusion-official-seq512-offset'+str(offset)]
            cases[name]=args
    for recompute in (False,True):
        name='small-key-fused'+('-recompute' if recompute else '')
        args=['launch-single.sh','--bf16','--all-gradients','--reference-profile','diagnostic',
              '--reference-core','fla','--aligned-reference','--shared-pointops',
              '--te-reference-norms','--te-final-reference','--precision-probe','key-fused',
              '--capture-output',name,'--result-tag=-key-fusion-checkpoint']
        if recompute:args+=['--recompute']
        cases[name]=args
    results={}
    for name,args in cases.items():
        if options.only and name not in options.only:continue
        command=['bash',str(AUDIT/args[0]),*args[1:]]
        log=ROOT/f'training-smoke/logs/audit-key-{name}.log'
        with log.open('w') as output:
            process=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        results[name]={'command':command,'log':str(log),'exit_code':process.returncode}
        (AUDIT/options.exits_file).write_text(json.dumps(results,indent=2)+'\n')
        print(name,'exit',process.returncode,flush=True)
        if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':main()
