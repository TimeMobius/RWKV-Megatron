"""Embedding control using real token IDs and a fixed upstream derivative."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from compare_layers import metric
ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(8)
    trace=torch.load(ROOT.parent/'cp-equivalence-20261021/cp1-full/trace-tp0-pp0-cp0.pt',weights_only=False)
    tokens=trace['batches'][0]['tokens'].cuda()
    # TP0's half of the vocabulary; exact production token mask.
    vocab=124160
    invalid=tokens>=vocab
    ids=tokens.masked_fill(invalid,0)
    torch.manual_seed(1234)
    grad=torch.randn((*tokens.shape,2304),device='cuda',dtype=torch.bfloat16)
    grad=grad.masked_fill(invalid[...,None],0)
    used=ids.unique()
    result={}
    for name,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
        weight=torch.zeros((vocab,2304),device='cuda',dtype=dtype,requires_grad=True)
        def backward(i,g):
            return torch.autograd.grad(F.embedding(i,weight),weight,g)[0].index_select(0,used).float()
        ref=backward(ids,grad.to(dtype))
        repeat=backward(ids,grad.to(dtype))
        parts=[backward(i.contiguous(),g.contiguous()) for i,g in zip(ids.chunk(2,1),grad.to(dtype).chunk(2,1))]
        result[name]={'repeat':metric(repeat,ref),'split':metric(parts[0]+parts[1],ref),'used_ids':used.numel()}
        del weight
    (ROOT/'embedding-grad.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
