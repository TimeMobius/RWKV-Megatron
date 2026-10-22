import copy
import io
import sys
from pathlib import Path

import torch
from transformer_engine.pytorch.optimizers import FusedAdam

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '3rdparty/Megatron-LM-0.11.0'))
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

options = dict(offload_fraction=0.5, cpu_optimizer_cls=torch.optim.AdamW,
               gpu_optimizer_cls=FusedAdam, param_update_in_fp32=True,
               overlap_cpu_optimizer_d2h_h2d=False, lr=1e-6, weight_decay=0.1,
               betas=(0.9, 0.995), eps=1e-8, fused=True)
params = [torch.nn.Parameter(torch.randn(65536, device='cuda', dtype=torch.bfloat16)) for _ in range(4)]
a = HybridDeviceOptimizer(params, **options)
for i in range(3):
    for p in params:
        p.decoupled_grad = torch.full_like(p, 0.125, dtype=torch.float32)
    a.step()
saved_params = [p.detach().clone() for p in params]
buffer = io.BytesIO()
torch.save(a.state_dict(), buffer)
buffer.seek(0)
saved = torch.load(buffer, map_location='cpu', weights_only=True)
bparams = [torch.nn.Parameter(p.clone()) for p in saved_params]
b = HybridDeviceOptimizer(bparams, **options)
b.dummy_step()
b.load_state_dict(copy.deepcopy(saved))
# Emulate the separate DistributedOptimizer tensor-state loading pass.
for p, state in b.state.items():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cpu().clone()
b._sync_hdo_state_to_sub_optimizers()
for sub in b.sub_optimizers:
    for p, state in sub.state.items():
        print(type(sub).__name__, str(p.device), tuple(p.shape),
              {k:(str(v.device), str(v.dtype),tuple(v.shape)) for k,v in state.items() if isinstance(v,torch.Tensor)}, flush=True)
for pa, pb in zip(params, bparams):
    pa.decoupled_grad = torch.full_like(pa, 0.125, dtype=torch.float32)
    pb.decoupled_grad = torch.full_like(pb, 0.125, dtype=torch.float32)
a.step()
b.step()
torch.cuda.synchronize()
for pa, pb in zip(params, bparams):
    torch.testing.assert_close(a.param_to_inner_param[pa], b.param_to_inner_param[pb], rtol=0, atol=0)
print('resume exact', flush=True)
