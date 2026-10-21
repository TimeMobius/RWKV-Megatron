"""Compare equal-TP/PP captures; CP ranks concatenate in contiguous token order."""
import argparse
import json
import math
from pathlib import Path

import torch

torch.set_num_threads(8)
HERE = Path(__file__).resolve().parent


def metric(actual, expected):
    a, r = actual.float(), expected.float()
    d = a-r
    rn, an, dn = float(r.norm()), float(a.norm()), float(d.norm())
    dot = float((a*r).sum())
    alpha = dot / max(rn*rn, 1e-30)
    return {'max_abs': float(d.abs().max()), 'relative_l2': dn/max(rn, 1e-30),
            'reference_norm': rn, 'actual_norm': an, 'diff_norm': dn,
            'scale_fit': alpha, 'dot': dot, 'finite': bool(a.isfinite().all() and r.isfinite().all())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,default=HERE/'cp1-uneven')
    parser.add_argument('--actual',type=Path,default=HERE/'cp2-uneven')
    parser.add_argument('--output',type=Path,default=HERE/'comparison-uneven.json')
    args=parser.parse_args()
    result={'reference':str(args.reference),'actual':str(args.actual),'inputs_equal':True,
            'forward':{},'gradients':{},'limits':{'gradient_global_relative_l2':0.03,
                'gradient_scale_fit_abs_error':0.02,'logits_relative_l2':0.02,
                'token_loss_max_abs':0.05}}
    for pp in range(2):
        for tp in range(2):
            key=f'tp{tp}-pp{pp}'
            ref=torch.load(args.reference/f'trace-{key}-cp0.pt',weights_only=True)
            shards=[torch.load(args.actual/f'trace-{key}-cp{i}.pt',weights_only=True) for i in range(2)]
            assert len(ref['batches'])==len(shards[0]['batches'])==len(shards[1]['batches'])
            for mb,batch in enumerate(ref['batches']):
                for name,value in batch.items():
                    if value is None:
                        assert all(s['batches'][mb][name] is None for s in shards)
                    else:
                        joined=torch.cat([s['batches'][mb][name] for s in shards],dim=1)
                        assert torch.equal(joined,value), (key,mb,name)
            if pp==1:
                for name in ('logits','head_inputs'):
                    assert len(ref[name])==len(shards[0][name])==len(shards[1][name])
                    for mb,expected in enumerate(ref[name]):
                        actual=torch.cat([s[name][mb] for s in shards],dim=0)
                        result['forward'][f'{key}-{name}-mb{mb}']=metric(actual,expected)
                for mb,expected in enumerate(ref['losses']):
                    actual=torch.cat([s['losses'][mb]['tokens'] for s in shards],dim=1)
                    row=metric(actual,expected['tokens'])
                    midpoint=actual.shape[1]//2
                    row['boundary_8_tokens']=metric(actual[:,midpoint-4:midpoint+4], expected['tokens'][:,midpoint-4:midpoint+4])
                    reports=[s['losses'][mb]['report'] for s in shards]
                    assert torch.equal(reports[0],reports[1])
                    row['reference_mean']=float(expected['report'][0]/expected['report'][1])
                    row['actual_mean']=float(reports[0][0]/reports[0][1])
                    row['mean_abs_diff']=abs(row['actual_mean']-row['reference_mean'])
                    row['reference_count']=expected['count_returned']
                    row['actual_count']=shards[0]['losses'][mb]['count_returned']
                    assert row['reference_count']==row['actual_count']
                    result['forward'][f'{key}-token_loss-mb{mb}']=row
            # mmap avoids reading all reference and actual gradients into RAM at once.
            refg=torch.load(args.reference/f'grad-{key}.pt',weights_only=True,mmap=True)
            actg=torch.load(args.actual/f'grad-{key}.pt',weights_only=True,mmap=True)
            assert set(refg)==set(actg)
            for name,expected in refg.items():
                result['gradients'][f'{key}:{name}']=metric(actg[name],expected)
            print('Compared',key,len(refg),'parameter tensors',flush=True)
            del refg,actg
    rows=list(result['gradients'].values())
    rn2=sum(row['reference_norm']**2 for row in rows)
    an2=sum(row['actual_norm']**2 for row in rows)
    dn2=sum(row['diff_norm']**2 for row in rows)
    dot=sum(row['dot'] for row in rows)
    result['gradient_aggregate']={'relative_l2':math.sqrt(dn2/max(rn2,1e-30)),
        'reference_norm_with_tp_replicas':math.sqrt(rn2),'actual_norm_with_tp_replicas':math.sqrt(an2),
        'norm_ratio':math.sqrt(an2/max(rn2,1e-30)),'scale_fit':dot/max(rn2,1e-30),
        'tensor_count':len(rows),'all_finite':all(row['finite'] for row in rows)}
    result['worst_gradients']=sorted([{'name':key,**row} for key,row in result['gradients'].items()],
                                     key=lambda row:row['relative_l2'],reverse=True)[:20]
    logits=[row for key,row in result['forward'].items() if '-logits-' in key]
    losses=[row for key,row in result['forward'].items() if '-token_loss-' in key]
    limits=result['limits']
    result['passed']=(result['gradient_aggregate']['all_finite'] and
        result['gradient_aggregate']['relative_l2']<limits['gradient_global_relative_l2'] and
        abs(result['gradient_aggregate']['scale_fit']-1)<limits['gradient_scale_fit_abs_error'] and
        all(row['finite'] and row['relative_l2']<limits['logits_relative_l2'] for row in logits) and
        all(row['finite'] and row['max_abs']<limits['token_loss_max_abs'] for row in losses))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'gradient_aggregate':result['gradient_aggregate'],
        'losses':losses,'logits':logits,'worst_gradients':result['worst_gradients'][:3]},indent=2),flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':main()
