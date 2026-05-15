"""
Runtime checks for RWKV custom parameter sequence-parallel synchronization tags.

These tests instantiate lightweight versions of the RWKV custom modules and
verify that replicated parameters which consume sequence-sharded activations are
marked with `param.sequence_parallel`, so Megatron will all-reduce their grads
across the tensor-parallel group during gradient finalization.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RWKV6_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv6_layer_specs.py"
RWKV7_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_layer_specs.py"


class _MinimalMegatronModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class _DummyColumnParallelLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _DummyRowParallelLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _dummy_set_tensor_model_parallel_attributes(param, **kwargs):
    param.tensor_model_parallel = kwargs.get("is_parallel", False)
    param.partition_dim = kwargs.get("dim", None)
    param.partition_stride = kwargs.get("stride", None)


def _base_namespace():
    return {
        "MegatronModule": _MinimalMegatronModule,
        "Tensor": torch.Tensor,
        "TransformerConfig": object,
        "Tuple": Tuple,
        "Optional": __import__("typing").Optional,
        "ShardedStateDict": dict,
        "torch": torch,
        "divide": lambda x, y: x // y,
        "set_tensor_model_parallel_attributes": _dummy_set_tensor_model_parallel_attributes,
        "get_tensor_model_parallel_world_size": lambda: 2,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_args": lambda: SimpleNamespace(transformer_impl="local"),
        "HAVE_TE": False,
        "ColumnParallelLinear": _DummyColumnParallelLinear,
        "RowParallelLinear": _DummyRowParallelLinear,
        "TEColumnParallelLinear": _DummyColumnParallelLinear,
        "TERowParallelLinear": _DummyRowParallelLinear,
        "make_sharded_tensors_for_checkpoint": lambda state_dict, prefix, sharded_map, sharded_offsets: state_dict,
        "einsum": lambda *args, **kwargs: None,
        "all_to_all_sp2hp": lambda x: x,
        "scatter_to_tensor_model_parallel_region": lambda x: x,
        "rearrange": lambda x, *args, **kwargs: x,
        "get_fp8_recipe": lambda: None,
        "te": SimpleNamespace(fp8_autocast=lambda **kwargs: __import__("contextlib").nullcontext()),
    }


def _load_selected_defs(path: Path, names: list[str]):
    source = path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(path))
    selected = [
        node
        for node in module_ast.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    selected_module = ast.Module(body=selected, type_ignores=[])
    code = compile(selected_module, str(path), "exec")
    namespace = _base_namespace()
    exec(code, namespace)
    return namespace


def _build_config(**overrides):
    base = dict(
        hidden_size=8,
        ffn_hidden_size=16,
        num_layers=4,
        num_attention_heads=2,
        sequence_parallel=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        init_method=None,
        output_layer_init_method=None,
        perform_initialization=False,
        use_cpu_initialization=False,
        layernorm_epsilon=1e-5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_rwkv7_channelmix_marks_mix_coeffs_sequence_parallel():
    namespace = _load_selected_defs(RWKV7_PATH, ["RWKV7ChannelMix"])
    module_cls = namespace["RWKV7ChannelMix"]
    config = _build_config()
    with patch("torch.cuda.current_device", lambda: "cpu"):
        module = module_cls(config)
    assert getattr(module.mix_coeffs, "sequence_parallel", False) is True


def test_rwkv6_channelmix_marks_mix_coeffs_sequence_parallel():
    namespace = _load_selected_defs(RWKV6_PATH, ["RWKV6ChannelMix"])
    module_cls = namespace["RWKV6ChannelMix"]
    config = _build_config()
    with patch("torch.cuda.current_device", lambda: "cpu"):
        module = module_cls(config)
    assert getattr(module.mix_coeffs, "sequence_parallel", False) is True


def test_rwkv7_bottleneck_marks_non_tp_bias_sequence_parallel():
    namespace = _load_selected_defs(RWKV7_PATH, ["RWKV7Bottleneck"])
    module_cls = namespace["RWKV7Bottleneck"]
    config = _build_config()
    with patch("torch.cuda.current_device", lambda: "cpu"):
        module = module_cls(config, bottleneck_size=4, to_tensor_parallel=False, use_bias=True)
    assert getattr(module.bias, "sequence_parallel", False) is True


def test_rwkv6_bottleneck_marks_non_tp_bias_sequence_parallel():
    namespace = _load_selected_defs(RWKV6_PATH, ["RWKV6Bottleneck"])
    module_cls = namespace["RWKV6Bottleneck"]
    config = _build_config()
    with patch("torch.cuda.current_device", lambda: "cpu"):
        module = module_cls(config, bottleneck_size=4, expansion=2, to_tensor_parallel=False)
    assert getattr(module.bias, "sequence_parallel", False) is True


def test_rwkv6_and_rwkv7_bottleneck_weights_stay_sequence_parallel():
    for path, class_name, kwargs in (
        (RWKV6_PATH, "RWKV6Bottleneck", dict(bottleneck_size=4, expansion=2, to_tensor_parallel=True)),
        (RWKV7_PATH, "RWKV7Bottleneck", dict(bottleneck_size=4, to_tensor_parallel=True, use_bias=True)),
    ):
        namespace = _load_selected_defs(path, [class_name])
        module_cls = namespace[class_name]
        config = _build_config()
        with patch("torch.cuda.current_device", lambda: "cpu"):
            module = module_cls(config, **kwargs)
        assert getattr(module.weight_up, "sequence_parallel", False) is True
        assert getattr(module.weight_down, "sequence_parallel", False) is True


if __name__ == "__main__":
    for name, value in list(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("PASS: RWKV sequence-parallel parameter tag runtime checks passed.")
