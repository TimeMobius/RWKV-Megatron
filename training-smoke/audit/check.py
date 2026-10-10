"""Analyze real distributed forward/backward captures without changing tolerances."""
import json
from pathlib import Path

import torch

PATH=Path(__file__).resolve().parent


def metric(actual, expected):
    actual,expected=actual.double(),expected.double()
    delta=actual-expected
    return {'max_abs':delta.abs().max().item(),
            'relative_l2':(delta.norm()/expected.norm().clamp_min(1e-12)).item(),
            'reference_l2':expected.norm().item(),'actual_l2':actual.norm().item(),
            'finite':bool(torch.isfinite(actual).all())}


def flatten_values(carrier):
    return carrier[1:].flatten(0,1)


def analyze(case,dense):
    gradients={name:metric(case['native_grads'][name],value)
               for name,value in dense['native_grads'].items()}
    actual=torch.cat([v.flatten().float() for v in case['native_grads'].values()])
    expected=torch.cat([dense['native_grads'][k].flatten().float() for k in case['native_grads']])
    report={'name':case['name'],'tp':case['tp'],'pp':case['pp'],'sp':case['sp'],
            'recompute':case['recompute'],'logits':metric(case['logits'],dense['logits']),
            'loss':metric(case['losses'],dense['losses']),
            'all_gradients':metric(actual,expected),'gradient_by_parameter':gradients,
            'worst_relative_gradient':max(gradients,key=lambda x:gradients[x]['relative_l2'])}
    local_v0=[];pp_checks=[]
    for rank in case['ranks']:
        captures=rank['captures']
        if 'output-0' in captures:
            for i,first in enumerate(captures['output-0']):
                value=flatten_values(first['value'])
                later=[]
                for key,rows in captures.items():
                    if key.startswith(('input-','output-')):
                        for row in rows:
                            if torch.equal(flatten_values(row['value']),value):later.append(key)
                local_v0.append({'tp':rank['tp'],'pp':rank['pp'],'sample':i,
                                 'unchanged_captures':sorted(set(later)),
                                 'v0_gradient_nonzero':bool(first['grad'][1:].float().norm()>0)
                                 if first['grad'] is not None else False})
        if 'pp-send' in captures:
            target=next(x for x in case['ranks'] if x['pp']==rank['pp']+1 and x['tp']==rank['tp'])
            received=target['captures']['pp-recv']
            for i,(sent,recv) in enumerate(zip(captures['pp-send'],received)):
                pp_checks.append({'tp':rank['tp'],'sample':i,'shape':list(sent['value'].shape),
                                  'forward_exact':torch.equal(sent['value'],recv['value']),
                                  'backward_exact':torch.equal(sent['grad'],recv['grad']),
                                  'v0_gradient_norm':sent['grad'][1:].float().norm().item()})
    report['v0_local']=local_v0;report['pp_transport']=pp_checks
    unchanged=0;v0_inputs=0
    for rank in case['ranks']:
        origin=next(x for x in case['ranks'] if x['pp']==0 and x['tp']==rank['tp'])
        original=origin['captures']['output-0']
        for key,rows in rank['captures'].items():
            if not key.startswith(('input-','output-')):continue
            for row in rows:
                v0_inputs+=1
                if any(torch.equal(flatten_values(row['value']),flatten_values(first['value']))
                       for first in original):unchanged+=1
    report['v0_unchanged_through_layers_and_pp']={'checked':v0_inputs,'unchanged':unchanged}

    # SP v0 is replicated on TP ranks; sum its partial gradient over those replicas.
    first_stage=[x for x in case['ranks'] if x['pp']==0]
    dense_capture=dense['ranks'][0]['captures']['output-0']
    v0=[]
    if first_stage and all('output-0' in x['captures'] for x in first_stage):
        for i,reference in enumerate(dense_capture):
            partials=[x['captures']['output-0'][i] for x in first_stage]
            grad=sum(flatten_values(row['grad']).float() for row in partials)
            v0.append({'sample':i,'forward':metric(flatten_values(partials[0]['value']),
                                                   flatten_values(reference['value'])),
                       'gradient':metric(grad,flatten_values(reference['grad']))})
    report['v0_vs_dense']=v0
    return report


def main():
    dense=torch.load(PATH/'dense.pth',weights_only=False)
    reports={}
    for path in PATH.glob('*.pth'):
        if path.stem in ('cropped-native','minipile-tokens','dense-detached','dense-fp32-detached') or path.stem.endswith(('dense','dense-fp32-naive')):continue
        case=torch.load(path,weights_only=False)
        reference=dense
        if 'fp32' in path.stem:reference=torch.load(PATH/'dense-fp32-naive.pth',weights_only=False)
        if path.stem.startswith('full-3b'):
            prefix='full-3b-minipile-' if 'minipile' in path.stem else 'full-3b-'
            reference=torch.load(PATH/(prefix+('dense-fp32-naive' if 'fp32' in path.stem else 'dense')+'.pth'),weights_only=False)
        if path.stem=='baseline':
            reports['old_vs_new']=dict(logits=metric(dense['logits'],case['logits']),
                                      loss=metric(dense['losses'],case['losses']),
                                      all_gradients=metric(torch.cat([dense['native_grads'][k].float().flatten() for k in case['native_grads']]),
                                                           torch.cat([v.float().flatten() for v in case['native_grads'].values()])))
            continue
        reports[path.stem]=analyze(case,reference)
    if (PATH/'dense-detached.pth').exists():
        detached=torch.load(PATH/'dense-detached.pth',weights_only=False)
        dense_rank=dense['ranks'][0]['captures'];detach_rank=detached['ranks'][0]['captures']
        ablation={'logits_exact':torch.equal(dense['logits'],detached['logits']),
                  'loss_exact':torch.equal(dense['losses'],detached['losses']),
                  'value_path_gradient':[], 'layer0_value_weight_gradient_change':
                  metric(dense['native_grads']['blocks.0.att.value.weight'],
                         detached['native_grads']['blocks.0.att.value.weight'])}
        for i,(a,b) in enumerate(zip(dense_rank['first-value'],detach_rank['first-value'])):
            difference=a['grad'].float()-b['grad'].float()
            via_v0=flatten_values(dense_rank['output-0'][i]['grad'])
            ablation['value_path_gradient'].append({'sample':i,'difference_vs_carrier_gradient':metric(difference,via_v0),
                                                    'via_v0_gradient_norm':via_v0.float().norm().item()})
        reports['v0_detach_ablation']=ablation
    (PATH/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
    for name,result in reports.items():
        print(name, {k:result[k] for k in ('logits','loss','all_gradients','worst_relative_gradient') if k in result})
        if 'pp_transport' in result:print('PP:',result['pp_transport'])
    print('v0 detach:',reports.get('v0_detach_ablation'))


if __name__=='__main__':main()
