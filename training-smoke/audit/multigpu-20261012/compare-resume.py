"""Exact continuation check across every TP/PP model shard and optimizer shard."""
import argparse
import gc
import json
from pathlib import Path

import torch
import importlib.util
spec=importlib.util.spec_from_file_location('checkpoint_compare',Path(__file__).resolve().parents[1]/'compare-single-resume.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
compare=module.compare

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
CHECKPOINTS=ROOT/'training-smoke/checkpoints'
torch.set_num_threads(4)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args();name=args.name
    left=CHECKPOINTS/f'multigpu-20261012-{name}-resume/iter_0000004'
    right=CHECKPOINTS/f'multigpu-20261012-{name}-continuous/iter_0000004'
    checks={}
    files=sorted(path.relative_to(left) for path in left.rglob('*.pt'))
    assert files and files==sorted(path.relative_to(right) for path in right.rglob('*.pt'))
    for rel in files:
        a=torch.load(left/rel,map_location='cpu',weights_only=False,mmap=True)
        b=torch.load(right/rel,map_location='cpu',weights_only=False,mmap=True)
        if rel.name=='model_optim_rng.pt':
            values={}
            for key in ('model','optimizer','opt_param_scheduler','rng_state','iteration'):
                assert key in a and key in b,(rel,key)
                values[key]=compare(a[key],b[key]);assert not values[key]['differences'],(rel,key,values[key])
        else:
            values=compare(a,b);assert not values['differences'],(rel,values)
        checks[str(rel)]=values
        del a,b;gc.collect()
        print('EXACT',rel,flush=True)
    events={}
    ranks=sorted(int(p.name.split('-rank')[1].split('-')[0]) for p in OUT.glob(f'{name}-continuous-rank*-events.json'))
    assert len(ranks)==8
    for rank in ranks:
        records={phase:json.loads((OUT/f'{name}-{phase}-rank{rank}-events.json').read_text()) for phase in ('save','resume','continuous')}
        train={phase:[r for r in rows if r['kind']=='train'] for phase,rows in records.items()}
        assert len(train['save'])==len(train['resume'])==2 and len(train['continuous'])==4
        assert train['save']+train['resume']==train['continuous'],(rank,train)
        evaluations={phase:[r for r in rows if r['kind']=='eval'] for phase,rows in records.items()}
        hashes=[r['batch_hashes'] for rows in evaluations.values() for r in rows]
        assert hashes and all(h==hashes[0] for h in hashes),(rank,hashes)
        for phase in ('save','resume'):
            for event in evaluations[phase]:
                reference=next(r for r in evaluations['continuous'] if r['iteration']==event['iteration'])
                assert event['losses']==reference['losses'],(rank,phase,event,reference)
        events[str(rank)]={'train_scalars_and_data_exact':True,'fixed_eval_samples_exact':True,'eval_loss_exact':True,'train':train['continuous'],'eval':evaluations['continuous']}
    result={'name':name,'checkpoint_files':checks,'ranks':events,'exact_continuation':True}
    (OUT/f'{name}-resume-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL CONTINUATION CHECKS EXACT',name)


if __name__=='__main__':main()
