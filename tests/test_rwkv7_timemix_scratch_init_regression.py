"""
Regression test for RWKV7 TimeMix scratch initialization.

Expected workflow:
1. Run this script before fixing rwkv7_layer_specs.py. It should fail because the
   RWKV7-specific scratch-init parameters are left at the sentinel values from torch.empty.
2. After fixing the initialization logic to match the official RWKV-7 formulas, this script
   should pass and confirm both the values and TP sharding are correct.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_layer_specs.py"
SENTINEL = 12345.0


class _MinimalMegatronModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def _sentinel_empty(*size, **kwargs):
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        shape = tuple(size[0])
    else:
        shape = tuple(size)
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", None)
    return torch.full(shape, SENTINEL, dtype=dtype, device=device)


def _load_rwkv7_timemix_class(tp_size: int, tp_rank: int, layer_offset: int):
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(SOURCE_PATH))

    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_rwkv7_")
    ]
    class_node = next(
        node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == "RWKV7TimeMix"
    )
    selected_module = ast.Module(body=[*helper_nodes, class_node], type_ignores=[])
    code = compile(selected_module, str(SOURCE_PATH), "exec")

    class _DummyBottleneck(torch.nn.Module):
        def __init__(self, config, bottleneck_size, to_tensor_parallel, activation_type="tanh", use_bias=True):
            super().__init__()
            local_hidden = config.hidden_size // tp_size if to_tensor_parallel else config.hidden_size
            self.weight_down = torch.nn.Parameter(torch.zeros(config.hidden_size, bottleneck_size))
            self.weight_up = torch.nn.Parameter(torch.zeros(bottleneck_size, config.hidden_size))
            self.bias = None if not use_bias else torch.nn.Parameter(torch.zeros(local_hidden))

    class _DummyColumnParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _DummyRowParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _DummyGroupNorm(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.bias = torch.nn.Parameter(torch.zeros(1))

    class _DummyWKV7(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    namespace = {
        "MegatronModule": _MinimalMegatronModule,
        "Tensor": torch.Tensor,
        "TransformerConfig": object,
        "Tuple": Tuple,
        "ShardedStateDict": dict,
        "torch": torch,
        "divide": lambda x, y: x // y,
        "RWKV7Bottleneck": _DummyBottleneck,
        "ColumnParallelLinear": _DummyColumnParallelLinear,
        "RowParallelLinear": _DummyRowParallelLinear,
        "TEColumnParallelLinear": _DummyColumnParallelLinear,
        "TERowParallelLinear": _DummyRowParallelLinear,
        "GroupNorm": _DummyGroupNorm,
        "WKV7": _DummyWKV7,
        "get_args": lambda: SimpleNamespace(transformer_impl="local"),
        "HAVE_TE": False,
        "te": SimpleNamespace(fp8_autocast=lambda **kwargs: None),
        "get_fp8_recipe": lambda: None,
        "get_tensor_model_parallel_world_size": lambda: tp_size,
        "get_tensor_model_parallel_rank": lambda: tp_rank,
        "get_pipeline_model_parallel_rank": lambda: 0,
        "set_tensor_model_parallel_attributes": lambda *args, **kwargs: None,
        "get_transformer_layer_offset": lambda config: layer_offset,
    }
    exec(code, namespace)
    return namespace["RWKV7TimeMix"]


def _build_config(hidden_size: int, num_heads: int, num_layers: int, tp_size: int):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_layers=num_layers,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        add_bias_linear=False,
        add_qkv_bias=False,
        init_method=None,
        output_layer_init_method=None,
        layernorm_epsilon=1e-5,
        D_AAA_LORA=4,
        D_DECAY_LORA=4,
        D_GATE_LORA=4,
        D_MV_LORA=4,
        perform_initialization=True,
        use_cpu_initialization=False,
        tensor_model_parallel_size=tp_size,
    )


def _official_timemix_init(hidden_size: int, num_heads: int, num_layers: int, layer_id: int):
    ddd = torch.arange(hidden_size, dtype=torch.float32) / hidden_size
    ratio_0_to_1 = layer_id / (num_layers - 1)
    ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)
    _ = ratio_0_to_1

    mix_coeff_input = torch.stack(
        [
            1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),
            1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0),
            1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0),
            1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0),
            1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0),
            1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),
        ]
    )

    linear = torch.arange(hidden_size, dtype=torch.float32) / (hidden_size - 1) - 0.5
    beta = torch.zeros(hidden_size, dtype=torch.float32) + 0.71 - linear * 0.1
    alpha = torch.zeros(hidden_size, dtype=torch.float32) + 1.02
    head_dim = hidden_size // num_heads
    rho = (torch.zeros(num_heads, head_dim, dtype=torch.float32) - 0.04).flatten()
    return mix_coeff_input, rho, beta, alpha


def _tp_shard(full_tensor: torch.Tensor, tp_size: int, tp_rank: int):
    shard_size = full_tensor.numel() // tp_size
    start = shard_size * tp_rank
    return full_tensor.narrow(0, start, shard_size).contiguous()


def _build_timemix(
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    layer_number: int,
    tp_size: int,
    tp_rank: int,
    layer_offset: int,
):
    config = _build_config(hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, tp_size=tp_size)
    rwkv7_time_mix = _load_rwkv7_timemix_class(tp_size=tp_size, tp_rank=tp_rank, layer_offset=layer_offset)
    with patch("torch.cuda.current_device", lambda: "cpu"):
        with patch("torch.empty", _sentinel_empty):
            module = rwkv7_time_mix(config=config, layer_number=layer_number)
    return module


def _assert_close(name: str, got: torch.Tensor, expected: torch.Tensor):
    assert torch.allclose(got, expected, atol=1e-6, rtol=0.0), (
        f"{name} mismatch\n"
        f"got:\n{got}\n"
        f"expected:\n{expected}"
    )


def test_timemix_scratch_init_matches_official_formula():
    hidden_size = 8
    num_heads = 2
    num_layers = 6
    layer_number = 3
    layer_id = layer_number - 1

    module = _build_timemix(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        layer_number=layer_number,
        tp_size=1,
        tp_rank=0,
        layer_offset=0,
    )
    expected_mix, expected_rho, expected_beta, expected_alpha = _official_timemix_init(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        layer_id=layer_id,
    )

    _assert_close("mix_coeff_input", module.mix_coeff_input.detach().cpu(), expected_mix)
    _assert_close("rho", module.rho.detach().cpu(), expected_rho)
    _assert_close("beta", module.beta.detach().cpu(), expected_beta)
    _assert_close("alpha", module.alpha.detach().cpu(), expected_alpha)


def test_timemix_scratch_init_uses_global_layer_id_under_pp():
    hidden_size = 8
    num_heads = 2
    num_layers = 8
    layer_number = 1
    layer_offset = 4
    tp_size = 2
    tp_rank = 1
    global_layer_id = layer_number + layer_offset - 1

    module = _build_timemix(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        layer_number=layer_number,
        tp_size=tp_size,
        tp_rank=tp_rank,
        layer_offset=layer_offset,
    )
    expected_mix, expected_rho, expected_beta, expected_alpha = _official_timemix_init(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        layer_id=global_layer_id,
    )

    _assert_close("pp mix_coeff_input", module.mix_coeff_input.detach().cpu(), expected_mix)
    _assert_close("pp rho", module.rho.detach().cpu(), _tp_shard(expected_rho, tp_size, tp_rank))
    _assert_close("pp beta", module.beta.detach().cpu(), _tp_shard(expected_beta, tp_size, tp_rank))
    _assert_close("pp alpha", module.alpha.detach().cpu(), _tp_shard(expected_alpha, tp_size, tp_rank))


if __name__ == "__main__":
    test_timemix_scratch_init_matches_official_formula()
    test_timemix_scratch_init_uses_global_layer_id_under_pp()
    print("PASS: RWKV7 TimeMix scratch-init matches the official RWKV-7 formulas.")
