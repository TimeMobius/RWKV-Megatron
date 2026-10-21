"""Same-device cost of the DPLR CP prefix-state kernel, including affine export."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import triton
from fla.ops.cp import chunk_delta_h
from fla.ops.generalized_delta_rule.dplr import chunk
from fla.ops.utils.constant import RCP_LN2

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = subprocess.check_output([
        'git', '-C', str(ROOT / '3rdparty/flash-linear-attention-260115'),
        'show', 'ff481ea6862c99a818858081571373694de7b4e7:fla/ops/cp/chunk_delta_h.py',
    ])
    path = OUT / 'baseline_chunk_delta_h.py'
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('baseline_chunk_delta_h', path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    trace = torch.load(ROOT / 'training-smoke/audit/cp-error-source-20261021/cp1/trace-tp0-pp0-cp0.pt', weights_only=False)
    inputs = []
    for name in ('r', 'w', 'k', 'v', 'a', 'b'):
        # Captured complete heads are independent; repeat to TP-local H=18.
        x = trace[f'0.input_{name}'][:2048].permute(1, 0, 2).reshape(1, 2048, 2, 64).cuda()
        inputs.append(x.repeat(1, 1, 9, 1).contiguous())
    r, gk, k, v, a, b = inputs
    gk = -gk
    cu = torch.tensor([0, 2048], device='cuda', dtype=torch.long)
    indices = chunk.prepare_chunk_indices(cu, 64)
    gi, ge = chunk.chunk_rwkv6_fwd_cumsum(gk, 64, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=indices)
    A_ab, _, A_ak, _, _, kg, ag, bg = chunk.chunk_dplr_fwd_intra(
        q=r, k=k, a=a, b=b, gi=gi, ge=ge, scale=1.,
        cu_seqlens=cu, safe_gate=True, chunk_size=64, chunk_indices=indices,
    )
    w, u, _ = chunk.prepare_wy_repr_fwd(
        ag=ag, A_ab=A_ab, A_ak=A_ak, v=v, cu_seqlens=cu,
        chunk_size=64, chunk_indices=indices, safe_gate=True,
    )
    hm = torch.empty((18, 64, 128), device='cuda', dtype=torch.float32)
    kw = dict(k=kg, v=v, w=w, g=None, gk=gi, bg=bg, u=u, hm=hm, cu_seqlens=cu,
              T=2048, H=18, HV=18, K=64, V=64, BT=64, BLOCK_SIZE=32, BK1=64, MULTI_SEQS=False)
    kernels = {'before': baseline.pre_process_fwd_kernel_merged, 'after': chunk_delta_h.pre_process_fwd_kernel_merged}
    outputs = {}
    for name, kernel in kernels.items():
        kernel[(4, 18)](**kw)
        outputs[name] = hm[..., :64].clone()
    _, _, reference = chunk.chunk_dplr_fwd_h(
        kg=kg, v=v, w=w, u=u, bg=bg, gk=gi,
        output_final_state=True, cu_seqlens=cu, chunk_size=64,
    )
    errors = {name: float((x.float()-reference[0]).norm()/reference.norm()) for name, x in outputs.items()}
    assert errors['after'] < 5e-5, errors
    timings = {name: [] for name in kernels}
    for repeat in range(3):
        order = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
        for name in order:
            kernel = kernels[name]
            timings[name].append(triton.testing.do_bench(lambda: kernel[(4, 18)](**kw), warmup=100, rep=300))
    median = {name: sorted(values)[1] for name, values in timings.items()}
    result = {
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'triton': triton.__version__,
        'baseline_fla_commit': 'ff481ea6862c99a818858081571373694de7b4e7',
        'shape': [1, 2048, 18, 64, 64], 'dtype': 'bfloat16', 'BT': 64, 'layout': 'varlen [0,2048]',
        'TRITON_F32_DEFAULT': os.environ.get('TRITON_F32_DEFAULT', '(unset)'),
        'prefix_relative_l2': errors, 'milliseconds': timings, 'median_milliseconds': median,
        'latency_ratio_after_before': median['after']/median['before'],
    }
    (OUT / 'benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
