"""Check actual single-GPU training continuation, model/optimizer/scheduler/RNG states."""
import json
import argparse
import re
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'training-smoke'
SUFFIX=''
PATTERN=re.compile(r'\] iteration\s+(\d+)/\s*\d+.*?consumed samples:\s*(\d+)'
 r'.*?learning rate:\s*([\d.Ee+-]+).*?lm loss:\s*([\d.Ee+-]+).*?grad norm:\s*([\d.Ee+-]+)'
 r'.*?skipped iterations:\s*(\d+).*?nan iterations:\s*(\d+)')


def rows(phase):
    result={}
    for line in (PATH/f'logs/single-training-{phase}{SUFFIX}.log').read_text().splitlines():
        m=PATTERN.search(line)
        if m:
            step,samples,lr,loss,grad,skipped,nan=m.groups()
            result[int(step)]={'samples':int(samples),'lr':float(lr),'loss':float(loss),
                               'grad_norm':float(grad),'skipped':int(skipped),'nan':int(nan)}
    return result


def compare(a,b,path='',summary=None):
    if summary is None:summary={'tensor_count':0,'tensor_elements':0,'differences':[]}
    if isinstance(a,torch.Tensor):
        summary['tensor_count']+=1;summary['tensor_elements']+=a.numel()
        equal=isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    elif isinstance(a,np.ndarray):equal=isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    elif isinstance(a,dict):
        equal=isinstance(b,dict) and a.keys()==b.keys()
        if equal:
            for key,value in a.items():compare(value,b[key],path+'/'+str(key),summary)
            return summary
    elif isinstance(a,(tuple,list)):
        equal=type(a)==type(b) and len(a)==len(b)
        if equal:
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),summary)
            return summary
    else:equal=type(a)==type(b) and a==b
    if not equal:summary['differences'].append(path)
    return summary


def main():
    global SUFFIX
    parser=argparse.ArgumentParser();parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--suffix')
    args=parser.parse_args();SUFFIX=args.suffix if args.suffix is not None else ('-fixed' if args.fixed else '')
    saved,resumed,continuous=[rows(x) for x in ('save','resume','continuous')]
    assert list(saved)==[1,2] and list(resumed)==[3,4] and list(continuous)==[1,2,3,4]
    assert saved|resumed==continuous,(saved,resumed,continuous)
    assert all(row['skipped']==row['nan']==0 for row in continuous.values())
    left=PATH/f'checkpoints/single-resume{SUFFIX}/iter_0000004/mp_rank_00'
    right=PATH/f'checkpoints/single-continuous{SUFFIX}/iter_0000004/mp_rank_00'
    a=torch.load(left/'model_optim_rng.pt',map_location='cpu',weights_only=False,mmap=True)
    b=torch.load(right/'model_optim_rng.pt',map_location='cpu',weights_only=False,mmap=True)
    checks={}
    for key in ('model','optimizer','opt_param_scheduler','rng_state','iteration'):
        assert key in a and key in b,key
        checks[key]=compare(a[key],b[key])
        if key!='rng_state':assert not checks[key]['differences'],(key,checks[key])
    files=sorted(f.name for f in left.glob('*') if f.name!='model_optim_rng.pt')
    for name in files:
        x=torch.load(left/name,map_location='cpu',weights_only=False,mmap=True)
        y=torch.load(right/name,map_location='cpu',weights_only=False,mmap=True)
        checks[name]=compare(x,y);assert not checks[name]['differences'],checks[name]
    result={'loss_and_grad_norm_exact_at_logged_precision':True,'rows':continuous,
            'checkpoint_comparisons':checks,'exact_model_optimizer_scheduler':True,
            'rng_exact':not checks['rng_state']['differences']}
    (PATH/f'audit/single-resume{SUFFIX}-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
