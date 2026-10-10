"""Frozen old/new FLA point-op replay against identical FP32 formulas."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--old', action='store_true')
parser.add_argument('--pin-backward', action='store_true')
args = parser.parse_args()
PATH = Path(__file__).resolve().parent
if args.old:
    sys.path.insert(0, str(PATH.parent / 'reference-repos/fla-79d12e8e'))

import torch
import triton
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7, addcmul_bwd_kernel1, fused_addcmul_fwd_kernel
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from fla.utils import assert_close


def main():
    torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.pin_backward:
        # Diagnostic only: same legal autotune config in the unchanged mix backward kernel.
        addcmul_bwd_kernel1.configs = [triton.Config({'BT': 4}, num_warps=4, num_stages=1)]
        addcmul_bwd_kernel1.cache.clear()
        addcmul_bwd_kernel1.cache_results = False
    result = {}
    tensors = {}
    for length in (128, 4096):
        shape = (1, length, 2560)
        x, delta = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        coeff = torch.rand(6, 2560, device='cuda', dtype=torch.bfloat16)
        dy = [torch.randn_like(x) for _ in range(6)]
        actual = [v.detach().clone().requires_grad_() for v in (x, delta, coeff)]
        expected = [v.detach().float().requires_grad_() for v in (x, delta, coeff)]
        out = fused_addcmul_rwkv7(actual[0], actual[1], *actual[2].unbind(0))
        ref = tuple(expected[0] + expected[1] * c for c in expected[2])
        grads = torch.autograd.grad(out, actual, dy)
        refgrads = torch.autograd.grad(ref, expected, [d.float() for d in dy])
        for i, (a, b) in enumerate(zip(out, ref)):
            assert_close(f'mix-output-{i}', a, b, ratio=0.002)
        # FLA coefficient gradients contain BF16 multiplication before FP32 reduction.
        # Use the upstream point-op numerical budget, including its gradient contract.
        for i, (a, b) in enumerate(zip(grads, refgrads)):
            assert_close(f'mix-gradient-{i}', a, b, ratio=0.002)
        tensors[f'mix-{length}'] = dict(output=[t.detach().cpu() for t in out],
                                       gradient=[t.detach().cpu() for t in grads])
        mix_config = {key: str(getattr(kernel, 'best_config', None))
                      for key, kernel in (('forward', fused_addcmul_fwd_kernel), ('backward', addcmul_bwd_kernel1))}
        k = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        a = torch.rand_like(k)
        ka = torch.rand(2560, device='cuda', dtype=torch.bfloat16)
        dk = torch.randn_like(k)
        actual = [v.detach().clone().requires_grad_() for v in (k, a, ka)]
        expected = [v.detach().float().requires_grad_() for v in (k, a, ka)]
        out = fused_k_rwkv7(*actual)
        ref = expected[0] * (1 + (expected[1] - 1) * expected[2])
        grads = torch.autograd.grad(out, actual, dk)
        refgrads = torch.autograd.grad(ref, expected, dk.float())
        assert_close('key-output', out, ref, ratio=0.002)
        for i, (g, r) in enumerate(zip(grads, refgrads)):
            assert_close(f'key-gradient-{i}', g, r, ratio=0.002)
        tensors[f'key-{length}'] = dict(output=out.detach().cpu(),
                                       gradient=[t.detach().cpu() for t in grads])
        result[str(length)] = dict(passed=True, shape=list(shape), mix_autotune=mix_config)
    suffix = ('old' if args.old else 'new') + ('-pinned' if args.pin_backward else '')
    torch.save(tensors, PATH / f'fla-pointops-{suffix}.pth')
    (PATH / f'fla-pointops-{suffix}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
