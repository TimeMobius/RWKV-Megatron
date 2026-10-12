"""All-gradient old/new equivalence and observed PP byte reductions."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from rwkv_megatron.value_residual import unpack_sharded_value

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('metrics',OUT.parent/'multigpu-20261012/analyze.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
torch.set_num_threads(4)


def load(tag):
    files=[p for p in (OUT/tag).glob('*.pth') if '-rank' not in p.stem and p.name!='minipile-tokens.pth']
    assert len(files)==1,(tag,files)
    return torch.load(files[0],map_location='cpu',weights_only=False,mmap=True)


def cache(row,rank,case):
    if case['sp'] and row.shape[0]==2:
        return unpack_sharded_value(row[1],case['tp'])
    full=row[1:].flatten(0,1)
    width=full.shape[-1]//case['tp']
    return full[:,:,rank['tp']*width:(rank['tp']+1)*width]


def compare(before,after):
    assert before['native_grads'].keys()==after['native_grads'].keys()
    params={k:metrics.metric(v,before['native_grads'][k]) for k,v in after['native_grads'].items()}
    total=sum(v['delta_squared'] for v in params.values());ref=sum(v['reference_squared'] for v in params.values())
    result={'logits':metrics.metric(after['logits'],before['logits']),'before_loss':before['losses'].tolist(),'after_loss':after['losses'].tolist(),
            'loss_max_delta':float((before['losses']-after['losses']).abs().max()),
            'all_gradients':{'relative_l2':(total/max(ref,1e-24))**.5,'parameters':len(params),'exact_parameters':sum(v['exact'] for v in params.values()),'all_finite':all(v['finite'] for v in params.values())},
            'gradient_by_parameter':params,'v0':[],'first_value':[],'pp':[]}
    for new in after['ranks']:
        old=next(r for r in before['ranks'] if r['tp']==new['tp'] and r['pp']==new['pp'])
        for key,rows in new['captures'].items():
            if key.startswith(('input-','output-')):
                assert len(rows)==len(old['captures'][key])
                for a,b in zip(rows,old['captures'][key]):
                    record={'key':key,'tp':new['tp'],'pp':new['pp'],'value':metrics.metric(cache(a['value'],new,after),cache(b['value'],old,before))}
                    if a['grad'] is not None and b['grad'] is not None:
                        record['gradient']=metrics.metric(cache(a['grad'],new,after),cache(b['grad'],old,before))
                    result['v0'].append(record)
        if 'first-value' in new['captures']:
            for a,b in zip(new['captures']['first-value'],old['captures']['first-value']):
                result['first_value'].append({'tp':new['tp'],'value':metrics.metric(a['value'],b['value']),
                    'gradient':metrics.metric(a['grad'],b['grad'])})
        if 'pp-send' in new['captures']:
            recv=next(r for r in after['ranks'] if r['tp']==new['tp'] and r['pp']==new['pp']+1)
            for a,b,r in zip(new['captures']['pp-send'],old['captures']['pp-send'],recv['captures']['pp-recv']):
                assert torch.equal(a['value'],r['value']) and torch.equal(a['grad'],r['grad'])
                result['pp'].append({'tp':new['tp'],'pp':new['pp'],'old_shape':list(b['value'].shape),'new_shape':list(a['value'].shape),
                   'old_bytes':b['value'].numel()*b['value'].element_size(),'new_bytes':a['value'].numel()*a['value'].element_size(),
                   'forward_exact':True,'backward_exact':True,'v0_grad_nonzero':bool(a['grad'][1:].count_nonzero())})
    assert result['all_gradients']['all_finite']
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',nargs='+',required=True,help='before tag; after tag derived by replacement');args=parser.parse_args()
    path=OUT/'precision.json';results=json.loads(path.read_text()) if path.exists() else {}
    for tag in args.pairs:
        result=compare(load(tag),load(tag.replace('-before-','-after-')))
        results[tag]=result
        print(tag,result['all_gradients'],result['logits']['relative_l2'],'lossdelta',result['loss_max_delta'],flush=True)
        (OUT/'precision.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
