"""Shared RWKV7 LayerNorm selection for embedding, layers and output."""
import warnings

from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.utils import is_te_min_version

try:
    from megatron.core.extensions.transformer_engine import HAVE_TE, TENorm
except ImportError:
    HAVE_TE = False
    TENorm = None


def get_rwkv7_layernorm_impl():
    if HAVE_TE and is_te_min_version("1.9.0"):
        return TENorm
    warnings.warn(
        "RWKV7 LayerNorm requires Transformer Engine >= 1.9.0; "
        "using Apex FusedLayerNorm for ln0, layer norms and ln_out.",
        RuntimeWarning,
    )
    return FusedLayerNorm
