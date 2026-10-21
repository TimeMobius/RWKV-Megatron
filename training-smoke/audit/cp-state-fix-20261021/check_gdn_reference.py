"""Reproduce the shared CP harness's ordinary GDN reference without NCCL teardown."""
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

torch.manual_seed(42)
q = F.normalize(torch.randn(1, 10240, 4, 64, device='cuda'), dim=-1).bfloat16().requires_grad_()
k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).bfloat16().requires_grad_()
v = torch.randn_like(q).requires_grad_()
g = F.logsigmoid(torch.randn(1, 10240, 4, device='cuda', dtype=torch.bfloat16)).requires_grad_()
beta = torch.randn_like(g, dtype=torch.float32).sigmoid().requires_grad_()
cu = torch.tensor([0, 3000, 7000, 10240], device='cuda', dtype=torch.long)
out, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu, state_v_first=False)
out.backward(torch.randn_like(out))
print('GDN non-CP reference passed', flush=True)
