"""RWKV7 FLA chunk kernel vs independent FP32 recurrence and torch autograd."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.ops.rwkv7 import chunk_rwkv7
from fla.utils import assert_close, device
from check import metric

PATH=Path(__file__).resolve().parent


def reference(r,w,k,v,a,b,h0):
    r,w,k,v,a,b=[x.transpose(1,2).float() for x in (r,w,k,v,a,b)]
    state=h0.float()
    values=[]
    for t in range(r.shape[2]):
        correction=(state*a[:,:,t].unsqueeze(-1)).sum(-2)
        state=state*w[:,:,t].exp().unsqueeze(-1)
        state=state+k[:,:,t].unsqueeze(-1)*v[:,:,t].unsqueeze(-2)
        state=state+b[:,:,t].unsqueeze(-1)*correction.unsqueeze(-2)
        values.append((state*r[:,:,t].unsqueeze(-1)).sum(-2))
    return torch.stack(values,dim=2).transpose(1,2),state


def main():
    torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for length in (64,128,129):
        shape=(2,length,2,64)
        r,k,v=[torch.randn(shape,device=device,dtype=torch.bfloat16)*0.25 for _ in range(3)]
        w=-(0.05+torch.rand(shape,device=device)*0.4).to(torch.bfloat16)
        normalized=F.normalize(torch.randn(shape,device=device),dim=-1)
        a=-normalized.to(torch.bfloat16)
        b=(normalized*torch.rand(shape,device=device)*0.2).to(torch.bfloat16)
        h0=torch.randn(2,2,64,64,device=device)*0.1
        ref=[x.detach().clone().requires_grad_() for x in (r,w,k,v,a,b,h0)]
        tri=[x.detach().clone().requires_grad_() for x in (r,w,k,v,a,b,h0)]
        ro,rh=reference(*ref)
        to,th=chunk_rwkv7(*tri[:6],initial_state=tri[6],output_final_state=True,
                         scale=1.,safe_gate=True,chunk_size=64)
        do=torch.randn_like(to).float();dh=torch.randn_like(th)
        ((ro*do).sum()+(rh*dh).sum()).backward()
        ((to.float()*do).sum()+(th*dh).sum()).backward()
        checks={'output':metric(to,ro),'final_state':metric(th,rh)}
        # Frozen thresholds from the upstream DPLR test: o 0.007; states/grads 0.008.
        assert_close('output',ro,to.float(),0.007)
        assert_close('state',rh,th,0.008)
        for name,x,y in zip(('r','w','k','v','a','b','h0'),ref,tri):
            assert_close('d'+name,x.grad.float(),y.grad.float(),0.008)
            checks['d'+name]=metric(y.grad,x.grad)
        rows.append({'batch':2,'sequence':length,'heads':2,'head_size':64,
                     'dtype':'bfloat16','safe_gate':True,'scale':1.,'chunk_size':64,
                     'checks':checks,'passed':True})
        print('KERNEL PASSED',length,{k:v['relative_l2'] for k,v in checks.items()},flush=True)
    (PATH/'kernel-results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
