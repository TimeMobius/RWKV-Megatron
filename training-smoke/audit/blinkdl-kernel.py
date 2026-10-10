"""Official clampw and FLA on identical raw inputs vs FP32 recurrence."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.ops.rwkv7 import chunk_rwkv7
from fla.utils import assert_close

from blinkdl_reference import fast_namespace
from check import metric
from kernel import reference

PATH=Path(__file__).resolve().parent


def main():
    namespace,_=fast_namespace()
    official=namespace['RWKV7_CLAMPW_CUDA']
    torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for length in (64,128):
        shape=(2,length,2,64)
        r,k,v=[torch.randn(shape,device='cuda',dtype=torch.bfloat16)*0.25 for _ in range(3)]
        wraw=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
        normalized=F.normalize(torch.randn(shape,device='cuda'),dim=-1)
        a=-normalized.to(torch.bfloat16)
        b=(normalized*torch.rand(shape,device='cuda')*0.2).to(torch.bfloat16)
        inputs=(r,wraw,k,v,a,b)
        oracle=[x.detach().clone().requires_grad_() for x in inputs]
        old=[x.detach().clone().requires_grad_() for x in inputs]
        new=[x.detach().clone().requires_grad_() for x in inputs]
        h0=torch.zeros(2,2,64,64,device='cuda')
        log_decay=-math.exp(-0.5)*oracle[1].float().sigmoid()
        ro,_=reference(oracle[0],log_decay,*oracle[2:],h0)
        flatten=lambda x:x.flatten(2).contiguous()
        oo=official(*map(flatten,old)).reshape(shape)
        # This is the production decay transformation, including BF16 rounding.
        new_decay=-(torch.exp(torch.tensor(-0.5))*new[1].sigmoid())
        no=chunk_rwkv7(new[0],new_decay,*new[2:],scale=1.,safe_gate=True,chunk_size=64)[0]
        upstream=torch.randn_like(ro)
        (ro*upstream).sum().backward()
        (oo.float()*upstream).sum().backward()
        (no.float()*upstream).sum().backward()
        checks={}
        failures=[]
        for name,output,tensors in [('official',oo,old),('fla',no,new)]:
            checks[name]={'output':metric(output,ro)}
            try:assert_close(name+'.output',ro,output.float(),0.007)
            except AssertionError as exc:failures.append(str(exc))
            for key,actual,expected in zip(('r','wraw','k','v','a','b'),tensors,oracle):
                checks[name]['d'+key]=metric(actual.grad,expected.grad)
                try:assert_close(name+'.d'+key,expected.grad.float(),actual.grad.float(),0.008)
                except AssertionError as exc:failures.append(str(exc))
        checks['fla_vs_official']={'output':metric(no,oo)}
        checks['fla_vs_official'].update({'d'+key:metric(actual.grad,expected.grad)
            for key,actual,expected in zip(('r','wraw','k','v','a','b'),new,old)})
        rows.append({'sequence':length,'batch':2,'heads':2,'head_size':64,
                     'thresholds':{'output':0.007,'gradients':0.008},
                     'checks':checks,'passed':not failures,'failures':failures})
    (PATH/'blinkdl-kernel-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('BLINKDL KERNEL',json.dumps(rows),flush=True)
    assert all(row['passed'] for row in rows), 'See preserved kernel differences; thresholds were not relaxed'


if __name__=='__main__':main()
