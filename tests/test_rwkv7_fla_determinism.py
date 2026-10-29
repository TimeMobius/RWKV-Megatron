"""Replay the production deterministic decay scan with an empty tuning cache."""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

from rwkv_megatron.fla_determinism import configure_rwkv7_determinism
from rwkv_megatron.precision import validate_rwkv_precision


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and FLA")
def test_production_policy_pins_and_replays_decay_scan():
    from triton.runtime.autotuner import Autotuner

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    module = importlib.import_module("fla.ops.generalized_delta_rule.dplr.chunk_A_bwd")
    kernel = module.chunk_dplr_bwd_dgk_kernel
    tuner = kernel
    while not isinstance(tuner, Autotuner):
        tuner = tuner.fn
    previous_configs = tuner.configs
    previous_cache = tuner.cache.copy()
    configure_rwkv7_determinism.cache_clear()
    try:
        validate_rwkv_precision(
            SimpleNamespace(fp8=None, deterministic_mode=True), have_te=False
        )
        assert len(tuner.configs) == 1
        config = tuner.configs[0]
        assert (config.kwargs, config.num_warps, config.num_stages) == (
            {"BK": 32},
            4,
            2,
        )
        assert not tuner.cache
        torch.manual_seed(1234)
        inputs = {
            "dgk": torch.randn(1, 128, 2, 64, device="cuda"),
            "dgk_offset": torch.randn(1, 128, 2, 64, device="cuda"),
            "dgk_last": torch.randn(1, 2, 2, 64, device="cuda"),
        }
        reference = None
        for _ in range(3):
            tuner.cache.clear()
            output = torch.empty_like(inputs["dgk"])
            kernel[(2, 2, 2)](
                **inputs,
                dgk_output=output,
                cu_seqlens=None,
                chunk_indices=None,
                T=128,
                H=2,
                K=64,
                BT=64,
            )
            if reference is None:
                reference = output.clone()
            else:
                assert torch.equal(reference, output)
    finally:
        tuner.configs = previous_configs
        tuner.cache.clear()
        tuner.cache.update(previous_cache)
        configure_rwkv7_determinism.cache_clear()
