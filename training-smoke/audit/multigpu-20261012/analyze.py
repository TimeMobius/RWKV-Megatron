"""Compare every exported gradient with bounded memory and record raw error metrics."""
import argparse
import json
from pathlib import Path

import torch

OUT=Path(__file__).resolve().parent
torch.set_num_threads(4)


def metric(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    af,bf=a.flatten(),b.flatten()
    sq=ref=actual=0.;mx=0.;finite=True;exact=torch.equal(a,b)
    for start in range(0,af.numel(),1048576):
        x,y=af[start:start+1048576].double(),bf[start:start+1048576].double()
        d=x-y;sq+=float(d.square().sum());ref+=float(y.square().sum());actual+=float(x.square().sum())
        mx=max(mx,float(d.abs().max()));finite=finite and bool(torch.isfinite(x).all())
    return dict(max_abs=mx,relative_l2=(sq/max(ref,1e-24))**.5,reference_l2=ref**.5,actual_l2=actual**.5,
                finite=finite,exact=exact,delta_squared=sq,reference_squared=ref)


def transport(case):
    checks=[];unchanged=[]
    first={x['tp']:x for x in case['ranks'] if x['pp']==0}
    for rank in case['ranks']:
        captures=rank['captures']
        if not case['recompute']:
            for key,rows in captures.items():
                if key.startswith(('input-','output-')):
                    for i,row in enumerate(rows):
                        origin=first[rank['tp']]['captures']['output-0'][i]
                        unchanged.append(torch.equal(row['value'][1:].flatten(0,1),origin['value'][1:].flatten(0,1)))
        if 'pp-send' in captures:
            target=next(x for x in case['ranks'] if x['tp']==rank['tp'] and x['pp']==rank['pp']+1)
            received=target['captures']['pp-recv']
            assert len(received)==len(captures['pp-send'])
            for i,(sent,recv) in enumerate(zip(captures['pp-send'],received)):
                checks.append(dict(tp=rank['tp'],pp=rank['pp'],microbatch=i,shape=list(sent['value'].shape),
                    forward_exact=torch.equal(sent['value'],recv['value']),
                    backward_exact=sent['grad'] is not None and recv['grad'] is not None and torch.equal(sent['grad'],recv['grad']),
                    v0_gradient_nonzero=sent['grad'] is not None and bool(sent['grad'][1:].count_nonzero())))
    return dict(pp_transport=checks,v0_carrier_unchanged=dict(checked=len(unchanged),exact=sum(unchanged)))


def traces(case,dense):
    rows={}
    if case['recompute'] or not case['ranks'][0].get('traces'):return rows
    sharded_hidden={'r','k-raw','v-raw','wkv','ffn-key'}
    sharded_sequence={'norm1','norm2','proj','ffn-value'}
    for source in dense['ranks']:
        for key,reference in source.get('traces',{}).items():
            ranks=sorted([x for x in case['ranks'] if key in x.get('traces',{})],key=lambda x:x['tp'])
            tag=key.split('.',1)[1]
            for i,y in enumerate(reference):
                if tag in sharded_hidden:x=torch.cat([r['traces'][key][i] for r in ranks],dim=-1)
                elif case['sp'] and tag in sharded_sequence:x=torch.cat([r['traces'][key][i] for r in ranks],dim=0)
                else:x=ranks[0]['traces'][key][i]
                rows[key+':'+str(i)]=metric(x,y)
    return rows


def compare(case,dense):
    assert case['native_grads'].keys()==dense['native_grads'].keys()
    params={k:metric(v,dense['native_grads'][k]) for k,v in case['native_grads'].items()}
    sq=sum(v['delta_squared'] for v in params.values());ref=sum(v['reference_squared'] for v in params.values())
    grad=dict(relative_l2=(sq/max(ref,1e-24))**.5,exact_parameters=sum(v['exact'] for v in params.values()),
              parameters=len(params),all_finite=all(v['finite'] for v in params.values()),
              worst=max(params,key=lambda k:params[k]['relative_l2']))
    v0=[]
    if not case['recompute'] and not dense['recompute']:
        ranks=[x for x in case['ranks'] if x['pp']==0]
        for i,r in enumerate(dense['ranks'][0]['captures']['output-0']):
            grad_sum=sum(x['captures']['output-0'][i]['grad'][1:].flatten(0,1).float() for x in ranks)
            v0.append(dict(value=metric(ranks[0]['captures']['output-0'][i]['value'][1:].flatten(0,1),r['value'][1:].flatten(0,1)),
                           gradient=metric(grad_sum,r['grad'][1:].flatten(0,1))))
    return dict(name=case['name'],reference=dense['name'],dimensions=case['dimensions'],
                logits=metric(case['logits'],dense['logits']),losses=case['losses'].tolist(),reference_losses=dense['losses'].tolist(),
                loss_max_delta=float((case['losses']-dense['losses']).abs().max()),all_gradients=grad,
                gradient_by_parameter=params,v0_vs_reference=v0,forward_trace_vs_dense=traces(case,dense),**transport(case))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',default='');parser.add_argument('--cases',nargs='+');args=parser.parse_args()
    result_path=OUT/f'{args.prefix}comparisons.json'
    results=json.loads(result_path.read_text()) if result_path.exists() else {}
    for name in args.cases:
        path=OUT/f'{args.prefix}{name}.pth'
        case=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        refname='dense-fp32-naive' if 'fp32' in name else 'dense'
        ref=torch.load(OUT/f'{args.prefix}{refname}.pth',map_location='cpu',weights_only=False,mmap=True)
        results[name]=compare(case,ref)
        if case['recompute']:
            noncheckpoint='tp4-sp-pp2' if name.startswith('tp4') else 'tp2-sp-pp2'
            if 'fp32' in name:noncheckpoint+='-fp32-naive'
            before=torch.load(OUT/f'{args.prefix}{noncheckpoint}.pth',map_location='cpu',weights_only=False,mmap=True)
            results[name]['vs_no_recompute']=compare(case,before)
        print(name,results[name]['all_gradients'],results[name]['logits'],flush=True)
        (OUT/f'{args.prefix}comparisons.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
