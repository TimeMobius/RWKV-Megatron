"""
Regression test for RWKV7 ChannelMix scratch initialization.

Expected workflow:
1. Run this script before fixing rwkv7_layer_specs.py. It should fail because
   mix_coeffs is left at the sentinel values from torch.empty.
2. After fixing the initialization logic to match the official RWKV-7 formula,
   this script should pass and confirm the layer-dependent x_k initialization.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import torch

from rwkv_megatron.precision import validate_rwkv_precision


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


def _load_rwkv7_channelmix_class():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(SOURCE_PATH))

    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_rwkv7_")
    ]
    class_node = next(
        node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == "RWKV7ChannelMix"
    )
    selected_module = ast.Module(body=[*helper_nodes, class_node], type_ignores=[])
    code = compile(selected_module, str(SOURCE_PATH), "exec")

    class _DummyColumnParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _DummyRowParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    namespace = {
        "MegatronModule": _MinimalMegatronModule,
        "Tensor": torch.Tensor,
        "TransformerConfig": object,
        "Tuple": Tuple,
        "torch": torch,
        "ColumnParallelLinear": _DummyColumnParallelLinear,
        "RowParallelLinear": _DummyRowParallelLinear,
        "TEColumnParallelLinear": _DummyColumnParallelLinear,
        "TERowParallelLinear": _DummyRowParallelLinear,
        "get_args": lambda: SimpleNamespace(transformer_impl="local"),
        "HAVE_TE": False,
        "validate_rwkv_precision": validate_rwkv_precision,
    }
    exec(code, namespace)
    return namespace["RWKV7ChannelMix"]


def _build_config(hidden_size: int, ffn_hidden_size: int, num_layers: int):
    return SimpleNamespace(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_layers=num_layers,
        sequence_parallel=False,
        add_bias_linear=False,
        init_method=None,
        output_layer_init_method=None,
        perform_initialization=True,
        use_cpu_initialization=False,
    )


def _official_channelmix_init(hidden_size: int, num_layers: int, layer_id: int):
    ddd = torch.arange(hidden_size, dtype=torch.float32) / hidden_size
    ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)
    return 1.0 - torch.pow(ddd, ratio_1_to_almost0**4)


def _build_channelmix(hidden_size: int, ffn_hidden_size: int, num_layers: int):
    config = _build_config(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_layers=num_layers,
    )
    rwkv7_channelmix = _load_rwkv7_channelmix_class()
    with patch("torch.cuda.current_device", lambda: "cpu"):
        with patch("torch.empty", _sentinel_empty):
            module = rwkv7_channelmix(config=config)
    return module


def _assert_close(name: str, got: torch.Tensor, expected: torch.Tensor):
    assert torch.allclose(got, expected, atol=1e-6, rtol=0.0), (
        f"{name} mismatch\n"
        f"got:\n{got}\n"
        f"expected:\n{expected}"
    )


def test_channelmix_scratch_init_matches_official_formula():
    hidden_size = 8
    ffn_hidden_size = 32
    num_layers = 6
    global_layer_number = 4
    layer_id = global_layer_number - 1

    module = _build_channelmix(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_layers=num_layers,
    )
    if hasattr(module, "set_layer_number"):
        module.set_layer_number(global_layer_number)

    expected = _official_channelmix_init(
        hidden_size=hidden_size,
        num_layers=num_layers,
        layer_id=layer_id,
    )
    _assert_close("mix_coeffs", module.mix_coeffs.detach().cpu(), expected)


if __name__ == "__main__":
    test_channelmix_scratch_init_matches_official_formula()
    print("PASS: RWKV7 ChannelMix scratch-init matches the official RWKV-7 formula.")
