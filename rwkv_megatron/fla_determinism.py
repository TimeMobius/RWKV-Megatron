"""Stabilize the arithmetic layout of RWKV7's FLA decay reverse scan."""

import importlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def configure_rwkv7_determinism():
    """Fix the process-wide FLA scan layout for Megatron deterministic mode.

    Autotuning BK/warps changes the FP32 reverse-scan addition tree and can
    change its final BF16 derivative. Use an existing supported FLA config so
    a restarted process uses the same arithmetic even with an empty cache.
    """
    from triton.runtime.autotuner import Autotuner

    module = importlib.import_module("fla.ops.generalized_delta_rule.dplr.chunk_A_bwd")
    tuner = module.chunk_dplr_bwd_dgk_kernel
    while not isinstance(tuner, Autotuner):
        if not hasattr(tuner, "fn"):
            raise RuntimeError(
                "Unsupported FLA decay scan wrapper for deterministic RWKV7"
            )
        tuner = tuner.fn
    candidates = [
        config
        for config in tuner.configs
        if config.kwargs == {"BK": 32}
        and config.num_warps == 4
        and config.num_stages == 2
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "FLA lacks the fixed decay scan config required by deterministic RWKV7"
        )
    tuner.configs = candidates
    tuner.cache.clear()
    logger.info("RWKV7 deterministic FLA decay reverse scan: %s", candidates[0])
