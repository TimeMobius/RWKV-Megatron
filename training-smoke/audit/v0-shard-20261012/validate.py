"""Accept lossless transport, FP32 formula parity, and bounded BF16 accumulation."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
VALUE_UPSTREAM={'emb.weight','blocks.0.ln0.weight','blocks.0.ln0.bias','blocks.0.ln1.weight',
                'blocks.0.ln1.bias','blocks.0.att.x_v','blocks.0.att.value.weight'}


def main():
    rows=json.loads((OUT/'precision.json').read_text())
    assert len(rows)==15,len(rows)
    checks={}
    for tag,r in rows.items():
        assert r['logits']['exact'] and r['loss_max_delta']==0,tag
        params=r['gradient_by_parameter'];assert r['all_gradients']['all_finite']
        assert all(v['exact'] for k,v in params.items() if k not in VALUE_UPSTREAM),tag
        assert all(x['value']['exact'] for x in r['v0']),tag
        # Branch gradients through PP/carriers themselves are exact; only the
        # final BF16 addition into the first value projection can differ.
        assert all(x['gradient']['exact'] for x in r['v0'] if 'gradient' in x),tag
        bound=1e-5 if 'fp32' in tag else .005
        assert r['all_gradients']['relative_l2']<bound,(tag,r['all_gradients'])
        for p in r['pp']:
            assert p['forward_exact'] and p['backward_exact'] and p['v0_grad_nonzero']
            if p['new_shape'][0]!=p['old_shape'][0]:
                assert p['new_shape'][0]==2 and p['old_shape'][0] in (3,5)
                assert p['new_bytes']/p['old_bytes']==2/p['old_shape'][0]
        checks[tag]={k:r[k] for k in ('all_gradients','loss_max_delta','pp')}
    resume={name:json.loads((OUT/f'{name}-resume-validation.json').read_text())['exact_continuation'] for name in ('tp2-sp-pp2','tp4-sp-pp2')}
    assert all(resume.values())
    result={'exact_forward_and_loss':True,'exact_v0_carrier_value_and_gradient':True,
            'only_value_upstream_bf16_accumulation_differs':True,'precision':checks,'resume':resume,
            'bf16_aggregate_gradient_limit':.005,'fp32_aggregate_gradient_limit':1e-5,
            'coverage':'TP2/4, SP on/off, PP2/4, full and norm selective recompute, 3B BF16/FP32, same-topology Adam resume'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: old/new transport and precision, all PP sends/receives, and exact continuation.')


if __name__=='__main__':main()
