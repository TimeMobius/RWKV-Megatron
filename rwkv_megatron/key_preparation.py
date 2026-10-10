"""RWKV7 key preparation with FP32 intermediates and optional CUDA fusion.

Head normalization is local to each TP shard; no sequence or batch reduction
is performed. Outputs retain the projection dtype used by the WKV operator.
"""
import os
import warnings

import torch
import torch.nn.functional as F

try:
    from torch._dynamo.exc import BackendCompilerFailed, Unsupported
except ImportError:
    _COMPILATION_ERRORS = (ImportError,)
else:
    _COMPILATION_ERRORS = (ImportError, Unsupported, BackendCompilerFailed)


def key_fp32(key, learn, beta, alpha, head_size=64):
    # Keep FP64 useful for independent formula/gradient checks.
    dtype = torch.float64 if key.dtype == torch.float64 else torch.float32
    k, a = key.to(dtype), learn.to(dtype)
    kk = F.normalize(
        (k * beta.to(dtype)).reshape(*key.shape[:-1], -1, head_size), dim=-1
    ).reshape_as(k)
    rep = k * (1 + (a - 1) * alpha.to(dtype))
    return rep.to(key.dtype), (-kk).to(key.dtype), (a * kk).to(key.dtype)


_compiled_key = None
_fusion_unavailable = False


def prepare_key(key, learn, beta, alpha, head_size=64, fusion=None):
    """Use compiled CUDA arithmetic, or ordinary Torch FP32 as a fallback.

    An explicit fusion boolean (from YAML) overrides RWKV7_KEY_FUSION.
    If unset, RWKV7_KEY_FUSION=0 selects the fallback. Backend changes can
    change rounding.
    """
    global _compiled_key, _fusion_unavailable
    enabled = os.environ.get('RWKV7_KEY_FUSION', '1') != '0' if fusion is None else fusion
    if (not key.is_cuda or key.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or not enabled or _fusion_unavailable):
        return key_fp32(key, learn, beta, alpha, head_size)
    try:
        if _compiled_key is None:
            if not callable(getattr(torch, 'compile', None)):
                raise ImportError('torch.compile is unavailable')
            _compiled_key = torch.compile(key_fp32, fullgraph=True, dynamic=False)
        return _compiled_key(key, learn, beta, alpha, head_size)
    except _COMPILATION_ERRORS as error:
        # Only compiler availability/compilation errors are recoverable. Shape,
        # CUDA runtime, OOM and backward errors must propagate to the caller.
        _fusion_unavailable = True
        warnings.warn(
            f'RWKV7 key fusion unavailable ({type(error).__name__}); '
            'using Torch FP32 key preparation. Set rwkv_key_fusion=false or RWKV7_KEY_FUSION=0 to '
            'select this backend consistently across runs.', RuntimeWarning,
        )
        return key_fp32(key, learn, beta, alpha, head_size)
