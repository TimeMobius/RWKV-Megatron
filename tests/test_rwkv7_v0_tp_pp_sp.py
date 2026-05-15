"""
Regression test for RWKV7 v0 / v_first handling with tensor parallel + pipeline
parallel when sequence parallel is enabled.

This test checks the intended sequence-parallel contract:
1. Each TP rank carries only its local-sequence chunk across PP stages.
2. Each carried v0 cache slot keeps full hidden size for one TP rank's
   full-sequence cache after Megatron's SP all-gather inside ColumnParallelLinear.
3. Non-interleaved PP schedules should therefore use 1 + TP tensor slots:
   hidden states + one v0 cache per TP rank.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULES_PATH = (
    REPO_ROOT
    / "3rdparty"
    / "Megatron-LM-0.11.0"
    / "megatron"
    / "core"
    / "pipeline_parallel"
    / "schedules.py"
)

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
        "parallel_state": parallel_state,
        "ModelType": model_type,
    }
    exec(code, namespace)
    return namespace["get_tensor_shapes"], model_type


def stage0_build_v0_sp(gathered_full_sequence_full_hidden: torch.Tensor):
    """Emulate the SP first-layer branch in RWKV7TimeMix.forward()."""
    layer_number = 1
    pp_rank = 0
    v0 = None

    if v0 is None and layer_number == 1 and pp_rank == 0:
        v0 = gathered_full_sequence_full_hidden

    return v0


def transformer_layer_pack_sp(hidden_states: torch.Tensor, v0_per_tp_rank: list[torch.Tensor]):
    """Emulate TransformerLayer's SP pack protocol with one v0 cache slot per TP rank."""
    return torch.cat([hidden_states.unsqueeze(0), *[v.unsqueeze(0) for v in v0_per_tp_rank]], dim=0)


def timemix_unpack_sp(packed_hidden_states: torch.Tensor):
    """Emulate the SP unpack path entering RWKV7TimeMix.forward()."""
    assert packed_hidden_states.dim() == 4
    hidden_states = packed_hidden_states[0]
    v0 = packed_hidden_states[1:].reshape(
        -1,
        packed_hidden_states.shape[2],
        packed_hidden_states.shape[3],
    )
    return hidden_states, v0


def later_stage_mix_value_sp(
    local_value_shard: torch.Tensor,
    v0_full_sequence_full_hidden: torch.Tensor,
    tp_rank: int,
    *,
    layer_number: int,
    pp_rank: int,
):
    """Emulate the SP later-stage branch in RWKV7TimeMix.forward()."""
    assert v0_full_sequence_full_hidden.shape[1:] == local_value_shard.shape[1:]

    if pp_rank == 0 and layer_number == 1:
        raise ValueError("v0 is not None for the first layer")

    shard_size = HIDDEN // TP_SIZE
    start_idx = tp_rank * shard_size
    end_idx = start_idx + shard_size
    v0_shard = v0_full_sequence_full_hidden[:, :, start_idx:end_idx]

    vt = torch.ones_like(local_value_shard)
    mixed_value = torch.lerp(local_value_shard, v0_shard, vt)
    return mixed_value, v0_shard


def test_sp_non_interleaved_pipeline_shape_uses_one_v0_cache_slot_per_tp_rank():
    get_tensor_shapes, model_type = _load_get_tensor_shapes()
    config = SimpleNamespace(sequence_parallel=True, hidden_size=HIDDEN)

    tensor_shapes = get_tensor_shapes(
        rank=0,
        model_type=model_type.encoder_or_decoder,
        seq_length=8,
        micro_batch_size=2,
        decoder_seq_length=None,
        config=config,
        encoder_decoder_xattn=False,
    )

    assert tensor_shapes == [(1 + TP_SIZE, 4, 2, HIDDEN)]


def test_sp_transformer_layer_protocol_keeps_local_sequence_v0_contract():
    # Each TP rank owns a different local-sequence chunk. The stage-local hidden
    # state stays sequence-sharded, but v0 is materialized at full sequence length
    # on each TP rank after Megatron's SP all-gather inside ColumnParallelLinear.
    rank0_stage0_hidden = torch.tensor([[[11.0, 12.0, 13.0, 14.0]]])
    rank1_stage0_hidden = torch.tensor([[[21.0, 22.0, 23.0, 24.0]]])

    rank0_full_seq_v0_full_hidden = torch.tensor(
        [
            [[1.0, 2.0, 101.0, 102.0]],
            [[3.0, 4.0, 103.0, 104.0]],
        ]
    )
    rank1_full_seq_v0_full_hidden = rank0_full_seq_v0_full_hidden.clone()

    rank0_v0 = stage0_build_v0_sp(rank0_full_seq_v0_full_hidden)
    rank1_v0 = stage0_build_v0_sp(rank1_full_seq_v0_full_hidden)

    packed_rank0 = transformer_layer_pack_sp(
        rank0_stage0_hidden,
        [rank0_v0[0], rank0_v0[1]],
    )
    packed_rank1 = transformer_layer_pack_sp(
        rank1_stage0_hidden,
        [rank1_v0[0], rank1_v0[1]],
    )

    assert packed_rank0.shape == (1 + TP_SIZE, 1, 1, HIDDEN)
    assert packed_rank1.shape == (1 + TP_SIZE, 1, 1, HIDDEN)

    unpacked_hidden0, unpacked_v00 = timemix_unpack_sp(packed_rank0)
    unpacked_hidden1, unpacked_v01 = timemix_unpack_sp(packed_rank1)

    assert torch.equal(unpacked_hidden0, rank0_stage0_hidden)
    assert torch.equal(unpacked_hidden1, rank1_stage0_hidden)
    assert torch.equal(unpacked_v00, rank0_full_seq_v0_full_hidden)
    assert torch.equal(unpacked_v01, rank1_full_seq_v0_full_hidden)

    # Later PP stages still hold local-sequence hidden states, but their v0 cache
    # spans the full gathered sequence, so value should align on the full sequence
    # dimension after ColumnParallelLinear's SP all-gather.
    rank0_local_value = torch.tensor([[[9.0, 9.0]], [[7.0, 7.0]]])
    rank1_local_value = torch.tensor([[[8.0, 8.0]], [[6.0, 6.0]]])

    mixed0, shard0 = later_stage_mix_value_sp(
        rank0_local_value, unpacked_v00, tp_rank=0, layer_number=1, pp_rank=1
    )
    mixed1, shard1 = later_stage_mix_value_sp(
        rank1_local_value, unpacked_v01, tp_rank=1, layer_number=1, pp_rank=1
    )

    assert torch.equal(shard0, torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]))
    assert torch.equal(shard1, torch.tensor([[[101.0, 102.0]], [[103.0, 104.0]]]))
    assert torch.equal(mixed0, shard0)
    assert torch.equal(mixed1, shard1)

    repacked_rank0 = transformer_layer_pack_sp(unpacked_hidden0, [unpacked_v00[0], unpacked_v00[1]])
    repacked_rank1 = transformer_layer_pack_sp(unpacked_hidden1, [unpacked_v01[0], unpacked_v01[1]])

    assert repacked_rank0.shape == (1 + TP_SIZE, 1, 1, HIDDEN)
    assert repacked_rank1.shape == (1 + TP_SIZE, 1, 1, HIDDEN)
    assert torch.equal(repacked_rank0[1:], rank0_full_seq_v0_full_hidden)
    assert torch.equal(repacked_rank1[1:], rank1_full_seq_v0_full_hidden)


if __name__ == "__main__":
    test_sp_non_interleaved_pipeline_shape_uses_one_v0_cache_slot_per_tp_rank()
    test_sp_transformer_layer_protocol_keeps_local_sequence_v0_contract()
    print("PASS: SP TP+PP v0 contract stays self-consistent with one cache slot per TP rank.")
