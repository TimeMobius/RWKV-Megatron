"""Compare corresponding first-microbatch channels from the production runs."""
import json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent

def metric(x,y):
    x,y=x.double(),y.double()
    d=x-y
    return {'max_abs':d.abs().max().item(), 'rel_l2':(d.norm()/y.norm().clamp_min(1e-30)).item(), 'nonzero':int(torch.count_nonzero(d)), 'elements':d.numel()}

def main():
    torch.set_num_threads(8)
    result={}
    for tp in (0,1):
        for pp in (0,1):
            base=torch.load(ROOT/f'cp1/trace-tp{tp}-pp{pp}-cp0.pt',weights_only=False)
            halves=[torch.load(ROOT/f'cp2/trace-tp{tp}-pp{pp}-cp{c}.pt',weights_only=False) for c in (0,1)]
            for key,b in base.items():
                if key.endswith('weight'):
                    continue
                cp=torch.cat([d[key] for d in halves],dim=0)
                assert cp.shape==b.shape, (key,cp.shape,b.shape)
                mid=b.shape[0]//2
                result[f'tp{tp}.{key}']={label:metric(cp[s],b[s]) for label,s in [('full',slice(None)),('first',slice(0,mid)),('second',slice(mid,None)),('boundary',slice(mid,mid+64))]}
    (ROOT/'layers.json').write_text(json.dumps(result,indent=2)+'\n')
    for layer in range(24):
        def get(name):return result[f'tp0.{layer}.{name}']
        norm,inp,wkv,out=get('norm'),get('input_r'),get('wkv'),get('layer')
        print(f'{layer:2d}: norm={norm["full"]["rel_l2"]:.6g} r={inp["full"]["rel_l2"]:.6g} WKV={wkv["full"]["rel_l2"]:.6g} (first {wkv["first"]["rel_l2"]:.6g}, second {wkv["second"]["rel_l2"]:.6g}) layer={out["full"]["rel_l2"]:.6g}')
    for name in ('gemm.linear_key.input','gemm.linear_key.output','gemm.bottleneck_learn.input','gemm.bottleneck_learn.output'):
        print(name,result['tp0.'+name])

if __name__=='__main__':main()
