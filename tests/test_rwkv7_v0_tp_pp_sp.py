"""SP pipeline payload keeps full-sequence local heads in a single v0 slot."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from rwkv_megatron.value_residual import pack_sharded_value, unpack_sharded_value

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULES_PATH = REPO_ROOT / "3rdparty/Megatron-LM-0.11.0/megatron/core/pipeline_parallel/schedules.py"
TP_SIZE = 2
HIDDEN = 4


def _load_get_tensor_shapes(tp_size: int = TP_SIZE, cp_size: int = 1):
    source = SCHEDULES_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(SCHEDULES_PATH))
    function_node = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_tensor_shapes"
    )
    selected_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(selected_module)
    code = compile(selected_module, str(SCHEDULES_PATH), "exec")

    model_type = SimpleNamespace(
        encoder_and_decoder="encoder_and_decoder",
        encoder_or_decoder="encoder_or_decoder",
    )
    parallel_state = SimpleNamespace(
        get_context_parallel_world_size=lambda: cp_size,
        get_tensor_model_parallel_world_size=lambda: tp_size,
    )
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "parallel_state": parallel_state,
        "ModelType": model_type,
    }
    exec(code, namespace)
    return namespace["get_tensor_shapes"], model_type


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 3])
def test_pipeline_shape_reserves_two_slots(tp_size, cp_size, batch):
    get_shapes, _ = _load_get_tensor_shapes(tp_size, cp_size)
    config = SimpleNamespace(sequence_parallel=True, hidden_size=16,
                             variable_seq_lengths=False, rwkv7_value_residual=True)
    shapes = get_shapes(seq_length=16, micro_batch_size=batch, decoder_seq_length=None,
                        config=config, tp_group=SimpleNamespace(size=lambda: tp_size),
                        cp_group=SimpleNamespace(size=lambda: cp_size))
    assert shapes == [(2, 16 // (tp_size * cp_size), batch, 16)]


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("local_sequence", [1, 5])
def test_local_head_payload_preserves_tokens_samples_and_gradients(tp_size, batch, local_sequence):
    hidden = 16
    sequence = local_sequence * tp_size
    # Distinct values on every axis expose any token/sample/head interchange.
    full = torch.arange(sequence * batch * hidden, dtype=torch.float64).reshape(sequence, batch, hidden)
    for rank in range(tp_size):
        local = full[:, :, rank * (hidden // tp_size):(rank + 1) * (hidden // tp_size)].clone().requires_grad_()
        packed = pack_sharded_value(local, tp_size)
        assert packed.shape == (local_sequence, batch, hidden)
        assert packed.numel() == full.numel() // tp_size
        assert packed.data_ptr() == local.data_ptr()
        assert torch.equal(packed.flatten(), local.flatten())
        restored = unpack_sharded_value(packed, tp_size)
        assert torch.equal(restored, local)
        # Model the unchanged cache crossing multiple PP/layer boundaries.
        for _ in range(3):
            restored = unpack_sharded_value(pack_sharded_value(restored, tp_size), tp_size)
        weights = (local.detach() + 1).square()
        (restored * weights).sum().backward()
        assert torch.equal(local.grad, weights)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_multiple_value_branches_reach_the_original_projection(tp_size):
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(8, 3, 16, dtype=torch.float64, generator=gen)
    w = torch.randn(16, 16, dtype=torch.float64, generator=gen, requires_grad=True)
    value = x @ w
    full_loss = 0
    local_loss = 0
    for rank in range(tp_size):
        shard = value[:, :, rank * (16 // tp_size):(rank + 1) * (16 // tp_size)]
        carried = unpack_sharded_value(pack_sharded_value(shard, tp_size), tp_size)
        # Three downstream layers with distinct blend gates and incoming gradients.
        for _ in range(3):
            other = torch.randn(shard.shape, dtype=torch.float64, generator=gen)
            gate = torch.rand(shard.shape, dtype=torch.float64, generator=gen)
            incoming = torch.randn(shard.shape, dtype=torch.float64, generator=gen)
            full_loss = full_loss + (torch.lerp(other, shard, gate) * incoming).sum()
            local_loss = local_loss + (torch.lerp(other, carried, gate) * incoming).sum()
    reference, = torch.autograd.grad(full_loss, w, retain_graph=True)
    actual, = torch.autograd.grad(local_loss, w)
    torch.testing.assert_close(actual, reference, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("sp", [False, True])
def test_standard_gpt_pipeline_shape_is_unchanged(sp):
    get_shapes, _ = _load_get_tensor_shapes()
    config = SimpleNamespace(sequence_parallel=sp, hidden_size=16, variable_seq_lengths=False)
    shape = get_shapes(seq_length=8, micro_batch_size=3, decoder_seq_length=None,
                       config=config, tp_group=SimpleNamespace(size=lambda: 2),
                       cp_group=SimpleNamespace(size=lambda: 1))
    assert shape == [(4 if sp else 8, 3, 16)]
