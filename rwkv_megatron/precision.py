"""Validate RWKV precision settings before allocating the model."""

import inspect
import warnings
from functools import wraps

from .fla_determinism import configure_rwkv7_determinism


def _has_weight_gradient_override():
    from transformer_engine.common.recipe import DelayedScaling

    return "override_linear_precision" in inspect.signature(DelayedScaling).parameters


def keep_time_mix_bf16(forward):
    """Keep recurrent inputs out of FP8 in the conservative FFN-only mode."""

    @wraps(forward)
    def wrapped(self, *args, **kwargs):
        if self.config.fp8 and getattr(self.config, "rwkv_fp8_scope", "ffn") == "ffn":
            import transformer_engine.pytorch as te

            # Inherit Megatron's context for FFN; suppress it for the complete
            # TimeMix so r/k/v are not rounded to FP8 before the recurrence.
            autocast = getattr(te, "autocast", None) or te.fp8_autocast
            with autocast(enabled=False):
                return forward(self, *args, **kwargs)
        return forward(self, *args, **kwargs)

    return wrapped


def validate_rwkv_precision(config, *, have_te):
    scope = getattr(config, "rwkv_fp8_scope", "ffn")
    if scope not in ("ffn", "all"):
        raise ValueError("rwkv_fp8_scope must be 'ffn' or 'all'")
    fp8 = getattr(config, "fp8", None)
    if fp8:
        if fp8 not in ("hybrid", "e4m3"):
            raise ValueError(
                "RWKV fp8 must be 'hybrid' or 'e4m3'; use false to disable"
            )
        if not have_te or config.transformer_impl != "transformer_engine":
            raise ValueError(
                "RWKV FP8 requires Transformer Engine and transformer_impl: transformer_engine"
            )
        if scope == "ffn" and getattr(config, "fp8_param", False):
            raise ValueError(
                "FFN-only RWKV FP8 requires BF16 stored weights; disable fp8_param_gather"
            )
        if (
            not getattr(config, "fp8_wgrad", True)
            and not _has_weight_gradient_override()
        ):
            warnings.warn(
                "This TE version ignores fp8_wgrad=False; enabled FP8 linears still "
                "compute weight gradients with FP8 operands.",
                RuntimeWarning,
                stacklevel=2,
            )
    if any(
        getattr(config, flag, False)
        for flag in ("fp8_dot_product_attention", "fp8_multi_head_attention")
    ):
        raise ValueError(
            "RWKV WKV does not use TE attention; keep FP8 attention flags false"
        )
    if getattr(config, "deterministic_mode", False):
        configure_rwkv7_determinism()
