"""Exercise the actual RWKV-LM loader without importing GPU training modules."""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE_PATH = Path(__file__).resolve().parents[1] / "rwkv_megatron" / "rwkv7_model.py"


def _divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _load_from_rwkv_lm(tp_rank=0, tp_size=1):
    module = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    model_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "RWKV7Model"
    )
    method = next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "from_rwkv_lm"
    )
    selected = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(selected)
    namespace = {
        "torch": torch,
        "divide": _divide,
        "get_tensor_model_parallel_rank": lambda: tp_rank,
        "get_tensor_model_parallel_world_size": lambda: tp_size,
    }
    exec(compile(selected, str(SOURCE_PATH), "exec"), namespace)
    return namespace["from_rwkv_lm"]


def _fixture(tp_rank=0, tp_size=1, layers_start=0, num_heads=8, zero_centered_gamma=False):
    hidden_size = 32
    num_layers = 4
    num_local_layers = 2
    state_dict = {}

    def add(key, shape):
        tensor = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        state_dict[key] = tensor / 100 + len(state_dict)

    add("emb.weight", (16, hidden_size))
    add("head.weight", (16, hidden_size))
    for key in ["blocks.0.ln0.weight", "blocks.0.ln0.bias", "ln_out.weight", "ln_out.bias"]:
        add(key, (hidden_size,))
    for layer_index in range(num_layers):
        prefix = f"blocks.{layer_index}."
        for suffix in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"]:
            add(prefix + suffix, (hidden_size,))
        for channel in "rwkvag":
            add(prefix + "att.x_" + channel, (1, 1, hidden_size))
        for name in ["a", "w", "g", "v"]:
            if name == "v" and layer_index == 0:
                continue
            add(prefix + f"att.{name}1", (hidden_size, 3))
            add(prefix + f"att.{name}2", (3, hidden_size))
            if name != "g":
                add(prefix + f"att.{name}0", (1, 1, hidden_size))
        add(prefix + "att.r_k", (num_heads, hidden_size // num_heads))
        for suffix in ["k_k", "k_a"]:
            add(prefix + "att." + suffix, (1, 1, hidden_size))
        for name in ["receptance", "key", "value", "output"]:
            add(prefix + f"att.{name}.weight", (hidden_size, hidden_size))
        for suffix in ["weight", "bias"]:
            add(prefix + "att.ln_x." + suffix, (hidden_size,))
        add(prefix + "ffn.x_k", (1, 1, hidden_size))
        add(prefix + "ffn.key.weight", (64, hidden_size))
        add(prefix + "ffn.value.weight", (hidden_size, 64))

    parameters = []
    expected_copies = []

    def parameter(key, split_dim=None, flatten=False, norm=False):
        expected = state_dict[key]
        if flatten:
            expected = expected.flatten()
        if split_dim is not None:
            local_size = _divide(expected.shape[split_dim], tp_size)
            expected = expected.narrow(split_dim, tp_rank * local_size, local_size)
        if norm and zero_centered_gamma:
            expected = expected - 1
        destination = torch.full_like(expected, -123.0)
        parameters.append(destination)
        expected_copies.append((destination, expected))
        return destination

    def norm(prefix):
        return SimpleNamespace(
            weight=parameter(prefix + "weight", norm=True),
            bias=parameter(prefix + "bias"),
        )

    layers = []
    for layer_index in range(layers_start, layers_start + num_local_layers):
        prefix = f"blocks.{layer_index}."

        def bottleneck(name):
            result = SimpleNamespace(
                weight_down=parameter(prefix + f"att.{name}1"),
                weight_up=parameter(prefix + f"att.{name}2"),
            )
            if name != "g":
                result.bias = parameter(prefix + f"att.{name}0", split_dim=0, flatten=True)
            return result

        mix_coeff_input = torch.full((6, hidden_size), -123.0)
        parameters.append(mix_coeff_input)
        expected_copies.append((mix_coeff_input, torch.stack([
            state_dict[prefix + "att.x_" + channel].flatten() for channel in "rwkvag"
        ])))
        attention = SimpleNamespace(
            mix_coeff_input=mix_coeff_input,
            bottleneck_learn=bottleneck("a"),
            bottleneck_decay=bottleneck("w"),
            bottleneck_gate=bottleneck("g"),
            rho=parameter(prefix + "att.r_k", split_dim=0, flatten=True),
            beta=parameter(prefix + "att.k_k", split_dim=0, flatten=True),
            alpha=parameter(prefix + "att.k_a", split_dim=0, flatten=True),
            group_norm=SimpleNamespace(
                weight=parameter(prefix + "att.ln_x.weight", split_dim=0),
                bias=parameter(prefix + "att.ln_x.bias", split_dim=0),
            ),
        )
        if layer_index != 0:
            attention.bottleneck_v = bottleneck("v")
        for attribute, key, split_dim in [
            ("linear_receptance", "receptance", 0),
            ("linear_key", "key", 0),
            ("linear_value", "value", 0),
            ("linear_proj", "output", 1),
        ]:
            setattr(attention, attribute, SimpleNamespace(
                weight=parameter(prefix + f"att.{key}.weight", split_dim=split_dim)
            ))
        layers.append(SimpleNamespace(
            input_layernorm=norm(prefix + "ln1."),
            pre_mlp_layernorm=norm(prefix + "ln2."),
            self_attention=attention,
            mlp=SimpleNamespace(
                mix_coeffs=parameter(prefix + "ffn.x_k", flatten=True),
                linear_up=SimpleNamespace(weight=parameter(prefix + "ffn.key.weight", split_dim=0)),
                linear_down=SimpleNamespace(weight=parameter(prefix + "ffn.value.weight", split_dim=1)),
            ),
        ))
    model = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=hidden_size, num_attention_heads=num_heads,
            layernorm_zero_centered_gamma=zero_centered_gamma,
        ),
        decoder=SimpleNamespace(layers=layers),
        pre_process=layers_start == 0,
        post_process=layers_start + num_local_layers == num_layers,
        _get_layers_start=lambda: layers_start,
    )
    if model.pre_process:
        model.embedding = SimpleNamespace(word_embeddings=SimpleNamespace(
            weight=parameter("emb.weight", split_dim=0)
        ))
        model.pre_norm = norm("blocks.0.ln0.")
    if model.post_process:
        model.decoder.final_layernorm = norm("ln_out.")
        model.output_layer = SimpleNamespace(weight=parameter("head.weight", split_dim=0))
    return model, state_dict, parameters, expected_copies


class TestRWKVLMHeadLayoutGuard(unittest.TestCase):
    def test_matching_layout_loads_all_parameters_across_tp_and_pp(self):
        for layers_start in [0, 2]:
            for tp_size in [1, 2, 4]:
                for tp_rank in range(tp_size):
                    with self.subTest(layers_start=layers_start, tp_size=tp_size, tp_rank=tp_rank):
                        model, state_dict, _, expected_copies = _fixture(tp_rank, tp_size, layers_start)
                        _load_from_rwkv_lm(tp_rank, tp_size)(model, state_dict, False)
                        for actual, expected in expected_copies:
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_matching_nondefault_head_size_is_not_hardcoded(self):
        model, state_dict, _, expected_copies = _fixture(num_heads=4)
        _load_from_rwkv_lm()(model, state_dict, False)
        for actual, expected in expected_copies:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_matching_zero_centered_gamma_loader_behavior_is_preserved(self):
        model, state_dict, _, expected_copies = _fixture(zero_centered_gamma=True)
        _load_from_rwkv_lm()(model, state_dict, False)
        for actual, expected in expected_copies:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_layout_fails_before_any_parameters_are_copied(self):
        for shape in [(4, 8), (32,), (1, 8, 4), (8, 5)]:
            for layers_start in [0, 2]:
                for tp_rank in range(4):
                    with self.subTest(shape=shape, layers_start=layers_start, tp_rank=tp_rank):
                        model, state_dict, parameters, _ = _fixture(tp_rank, 4, layers_start)
                        key = f"blocks.{layers_start + 1}.att.r_k"
                        state_dict[key] = torch.zeros(shape)
                        with self.assertRaises(ValueError) as caught:
                            _load_from_rwkv_lm(tp_rank, 4)(model, state_dict, False)
                        message = str(caught.exception)
                        self.assertIn(key, message)
                        self.assertIn(str(shape), message)
                        self.assertIn("(8, 4)", message)
                        self.assertIn("num_attention_heads=8", message)
                        for parameter in parameters:
                            torch.testing.assert_close(parameter, torch.full_like(parameter, -123.0), rtol=0, atol=0)

    def test_actual_30b_head_mismatch_is_rejected(self):
        model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=6144, num_attention_heads=64, layernorm_zero_centered_gamma=False),
            decoder=SimpleNamespace(layers=[object()]),
            _get_layers_start=lambda: 32,
        )
        with self.assertRaises(ValueError) as caught:
            _load_from_rwkv_lm(tp_size=4)(model, {"blocks.32.att.r_k": torch.zeros(96, 64)}, False)
        message = str(caught.exception)
        self.assertIn("blocks.32.att.r_k", message)
        self.assertIn("(96, 64)", message)
        self.assertIn("(64, 96)", message)
        self.assertIn("num_attention_heads=64", message)

    def test_unsupported_reference_comparison_still_fails_first(self):
        with self.assertRaisesRegex(NotImplementedError, "rwkv-huggingface"):
            _load_from_rwkv_lm()(object(), {}, True)


if __name__ == "__main__":
    unittest.main()
