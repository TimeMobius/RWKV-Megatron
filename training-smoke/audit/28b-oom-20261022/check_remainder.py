"""Compare TE remainder representation against full FP32 masters on fixed gradients."""
import json
from pathlib import Path

import torch
from transformer_engine.pytorch.optimizers import FusedAdam

out = Path(__file__).resolve().parent
with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(1234)
    a = [torch.nn.Parameter(torch.randn(65536, device='cuda', dtype=torch.bfloat16) * scale)
         for scale in [0.001, 1.0, 100.0]]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    opts = dict(lr=1e-6, betas=(0.9, 0.995), eps=1e-8, weight_decay=0.1,
                master_weights=True, use_decoupled_grad=True, master_weight_dtype=torch.float32,
                exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32)
    oa = FusedAdam(a, store_param_remainders=False, **opts)
    ob = FusedAdam(b, store_param_remainders=True, **opts)
    comparisons=[]
    for step in range(100):
        for pa, pb in zip(a,b):
            grad = torch.randn_like(pa, dtype=torch.float32)
            pa.decoupled_grad = grad
            pb.decoupled_grad = grad
        oa.step()
        ob.step()
        for index,(pa, pb) in enumerate(zip(a,b)):
            x,y=pa.detach().float(),pb.detach().float()
            comparisons.append(dict(step=step+1,param=index,unequal_elements=int((x!=y).sum().item()),
                max_abs_delta=(x-y).abs().max().item(),
                relative_l2=((x-y).norm()/x.norm().clamp_min(1e-30)).item()))
            for key in ['exp_avg','exp_avg_sq']:
                torch.testing.assert_close(oa.state[pa][key],ob.state[pb][key],rtol=0,atol=0)
    result = dict(steps=100,parameters_compared=196608,compute_weights_bitwise_equal=all(c["unequal_elements"]==0 for c in comparisons),
                  first_compute_difference=next((c for c in comparisons if c["unequal_elements"]),None),
                  max_compute_relative_l2=max(c["relative_l2"] for c in comparisons),
                  max_unequal_elements_per_parameter=max(c["unequal_elements"] for c in comparisons),
                  adam_moments_bitwise_equal=True,
                  full_master_dtypes=[str(oa.state[p]['master_param'].dtype) for p in a],
                  remainder_dtypes=[str(ob.state[p]['master_param'].dtype) for p in b])
    (out/'remainder-parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
