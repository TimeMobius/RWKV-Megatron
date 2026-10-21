"""Check unchanged GDN/KDA/GDN2 prefix branches against the original kernel."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged

OUT = Path(__file__).resolve().parent


def main():
    torch.manual_seed(42)
    path = OUT / 'baseline_chunk_delta_h.py'
    spec = importlib.util.spec_from_file_location('baseline_shared_prefix', path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    B, T, H, HV, K, V, BT = 1, 192, 2, 4, 64, 32, 64
    k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, T, HV, K, device='cuda', dtype=torch.bfloat16) * 0.1
    u = torch.randn(B, T, HV, V, device='cuda', dtype=torch.bfloat16)
    hm = torch.empty((HV, K, V+K), device='cuda', dtype=torch.float32)
    results = {}
    for mode in ('g', 'gk', 'none'):
        g = gk = None
        if mode == 'g':
            g = (-0.01 * torch.rand(B, T//BT, BT, HV, device='cuda')).cumsum(2).view(B, T, HV)
        if mode == 'gk':
            gk = (-0.01 * torch.rand(B, T//BT, BT, HV, K, device='cuda')).cumsum(2).view(B, T, HV, K)
        for varlen in (False, True):
            cu = torch.tensor([0, T], device='cuda', dtype=torch.long) if varlen else None
            outputs = []
            for kernel in (baseline.pre_process_fwd_kernel_merged, pre_process_fwd_kernel_merged):
                hm.fill_(float('nan'))
                kernel[(3, HV)](k=k, v=u, w=w, g=g, gk=gk, bg=None, u=u, hm=hm, cu_seqlens=cu,
                                 T=T, H=H, HV=HV, K=K, V=V, BT=BT, BLOCK_SIZE=32, BK1=64, MULTI_SEQS=False)
                outputs.append(hm.clone())
            assert all(torch.isfinite(x).all() for x in outputs), (mode, varlen)
            assert torch.equal(*outputs), (mode, varlen)
            results[f'{mode}.varlen{varlen}'] = {'bitwise_equal': True, 'shape': list(hm.shape)}
    (OUT / 'shared-prefix.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
