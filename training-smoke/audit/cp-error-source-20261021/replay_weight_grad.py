"""Fixed-input/derivative GEMM control: BF16 local wgrad before FP32 DDP sum."""
import json
from pathlib import Path
import torch
from compare_layers import metric
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    trace=torch.load(ROOT/'cp1/trace-tp0-pp0-cp0.pt',weights_only=False)
    x=trace['gemm.bottleneck_learn.input'].cuda().squeeze(1)
    w=trace['gemm.bottleneck_learn.weight'].cuda()
    torch.manual_seed(1234)
    dy=torch.randn((x.shape[0],w.shape[1]),device='cuda',dtype=torch.bfloat16)
    result={}
    for name,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
        a,b=x.to(dtype).detach().requires_grad_(),w.to(dtype).detach().requires_grad_()
        out=a@b
        dx,dw=torch.autograd.grad(out,(a,b),dy.to(dtype))
        pairs=[]
        outputs=[]
        for ah,dh in zip(a.detach().chunk(2),dy.to(dtype).chunk(2)):
            ah=ah.contiguous().requires_grad_()
            wh=b.detach().clone().requires_grad_()
            oh=ah@wh
            pairs.append(torch.autograd.grad(oh,(ah,wh),dh.contiguous()))
            outputs.append(oh)
        # Production accumulates local BF16 parameter gradients into FP32
        # buffers, then CP all-reduces/reduce-scatters those buffers.
        result[name]={'forward':metric(torch.cat(outputs),out),
                      'dgrad':metric(torch.cat([p[0] for p in pairs]),dx),
                      'wgrad':metric(pairs[0][1].float()+pairs[1][1].float(),dw.float())}
    (ROOT/'weight-grad.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
