"""
Regression test for RWKV7TimeMix distributed-checkpoint sharding metadata.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import torch

from rwkv_megatron.precision import keep_time_mix_bf16, validate_rwkv_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_layer_specs.py"


class _MinimalMegatronModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def _load_rwkv7_timemix(tp_size: int = 2):
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

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            return {f"{prefix}weight": ("column",)}

    class _DummyRowParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            return {f"{prefix}weight": ("row",)}

    class _DummyWKV7(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            return {}

    calls = []

    def _capture_make_sharded_tensors_for_checkpoint(
        state_dict, prefix, tensor_parallel_layers_axis_map=None, sharded_offsets=()
    ):
        calls.append(
            {
                "prefix": prefix,
                "keys": tuple(state_dict.keys()),
                "tp_axis_map": dict(tensor_parallel_layers_axis_map or {}),
                "sharded_offsets": tuple(sharded_offsets),
            }
        )
        return {f"{prefix}{key}": value for key, value in state_dict.items()}

    def _sharded_state_dict_default(module, prefix, sharded_offsets, metadata):
        if hasattr(module, "sharded_state_dict"):
            return module.sharded_state_dict(
                prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
            )
        return {}

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
        "keep_time_mix_bf16": keep_time_mix_bf16,
        "validate_rwkv_precision": validate_rwkv_precision,
        "get_tensor_model_parallel_world_size": lambda: tp_size,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_pipeline_model_parallel_rank": lambda: 0,
        "set_tensor_model_parallel_attributes": lambda *args, **kwargs: None,
        "make_sharded_tensors_for_checkpoint": _capture_make_sharded_tensors_for_checkpoint,
        "sharded_state_dict_default": _sharded_state_dict_default,
        "get_transformer_layer_offset": lambda config: 0,
    }
    exec(code, namespace)
    return namespace["RWKV7TimeMix"], calls


def _build_config(hidden_size: int = 768, num_heads: int = 12):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_layers=6,
        pipeline_model_parallel_size=1,
        sequence_parallel=True,
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


def test_timemix_sharded_state_dict_marks_local_tp_parameters():
    rwkv7_timemix_cls, calls = _load_rwkv7_timemix()
    config = _build_config()

    with patch("torch.cuda.current_device", lambda: "cpu"):
        module = rwkv7_timemix_cls(config=config, layer_number=1)

    sharded = module.sharded_state_dict(prefix="decoder.layers.0.self_attention.")

    assert calls, "make_sharded_tensors_for_checkpoint should be called for local parameters"
    self_call = calls[0]
    assert self_call["prefix"] == "decoder.layers.0.self_attention."
    assert {"rho", "beta", "alpha"}.issubset(set(self_call["keys"]))
    assert self_call["tp_axis_map"] == {"rho": 0, "beta": 0, "alpha": 0}

    assert "decoder.layers.0.self_attention.rho" in sharded
    assert "decoder.layers.0.self_attention.beta" in sharded
    assert "decoder.layers.0.self_attention.alpha" in sharded
    assert "decoder.layers.0.self_attention.linear_key.weight" in sharded


if __name__ == "__main__":
    test_timemix_sharded_state_dict_marks_local_tp_parameters()
    print("PASS: RWKV7TimeMix sharded_state_dict marks TP-local parameters.")
