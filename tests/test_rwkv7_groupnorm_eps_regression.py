"""
Regression test for RWKV7 GroupNorm epsilon handling.

Expected workflow:
1. Run this script before fixing rwkv7_layer_specs.py. It should fail because:
   - GroupNorm ignores the explicit eps argument
   - RWKV7TimeMix passes layernorm_epsilon instead of head_dim * 1e-5
2. After fixing both issues, this script should pass and confirm RWKV7 uses the
   official GroupNorm epsilon.
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


class _MinimalMegatronModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def _load_groupnorm_and_timemix(tp_size: int = 1):
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(SOURCE_PATH))

    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_rwkv7_")
    ]
    class_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.ClassDef) and node.name in {"GroupNorm", "RWKV7TimeMix"}
    ]
    selected_module = ast.Module(body=[*helper_nodes, *class_nodes], type_ignores=[])
    code = compile(selected_module, str(SOURCE_PATH), "exec")

    class _DummyBottleneck(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight_down = torch.nn.Parameter(torch.zeros(1))
            self.weight_up = torch.nn.Parameter(torch.zeros(1))
            self.bias = torch.nn.Parameter(torch.zeros(1))

    class _DummyColumnParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _DummyRowParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _DummyWKV7(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    namespace = {
        "MegatronModule": _MinimalMegatronModule,
        "Tensor": torch.Tensor,
        "TransformerConfig": object,
        "Tuple": Tuple,
        "Optional": __import__("typing").Optional,
        "ShardedStateDict": dict,
        "torch": torch,
        "divide": lambda x, y: x // y,
        "RWKV7Bottleneck": _DummyBottleneck,
        "ColumnParallelLinear": _DummyColumnParallelLinear,
        "RowParallelLinear": _DummyRowParallelLinear,
        "TEColumnParallelLinear": _DummyColumnParallelLinear,
        "TERowParallelLinear": _DummyRowParallelLinear,
        "WKV7": _DummyWKV7,
        "get_args": lambda: SimpleNamespace(transformer_impl="local"),
        "HAVE_TE": False,
        "te": SimpleNamespace(fp8_autocast=lambda **kwargs: None),
        "get_fp8_recipe": lambda: None,
        "get_tensor_model_parallel_world_size": lambda: tp_size,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_pipeline_model_parallel_rank": lambda: 0,
        "set_tensor_model_parallel_attributes": lambda *args, **kwargs: None,
        "make_sharded_tensors_for_checkpoint": lambda *args, **kwargs: {},
        "get_transformer_layer_offset": lambda config: 0,
    }
    exec(code, namespace)
    return namespace["GroupNorm"], namespace["RWKV7TimeMix"]


def _build_config(hidden_size: int, num_heads: int):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_layers=6,
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
        perform_initialization=False,
        use_cpu_initialization=False,
    )


def test_groupnorm_respects_explicit_eps_argument():
    group_norm_cls, _ = _load_groupnorm_and_timemix()
    config = _build_config(hidden_size=128, num_heads=2)
    with patch("torch.cuda.current_device", lambda: "cpu"):
        module = group_norm_cls(
            config=config,
            num_groups=config.num_attention_heads,
            num_channels=config.hidden_size,
            eps=0.1234,
        )
    assert module.eps == 0.1234, f"GroupNorm.eps should use explicit argument, got {module.eps}"


def test_timemix_uses_official_groupnorm_eps():
    _, rwkv7_timemix_cls = _load_groupnorm_and_timemix()
    config = _build_config(hidden_size=768, num_heads=12)
    expected_eps = (config.hidden_size // config.num_attention_heads) * 1e-5
    with patch("torch.cuda.current_device", lambda: "cpu"):
        module = rwkv7_timemix_cls(config=config, layer_number=1)
    assert module.group_norm.eps == expected_eps, (
        f"RWKV7TimeMix should use head_dim * 1e-5 = {expected_eps}, got {module.group_norm.eps}"
    )


if __name__ == "__main__":
    test_groupnorm_respects_explicit_eps_argument()
    test_timemix_uses_official_groupnorm_eps()
    print("PASS: RWKV7 GroupNorm epsilon matches the official RWKV-7 setting.")
