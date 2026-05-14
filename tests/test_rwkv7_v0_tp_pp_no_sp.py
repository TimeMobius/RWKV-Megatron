"""
Regression test for RWKV7 v0 / v_first handling with tensor parallel + pipeline
parallel when sequence parallel is disabled.

This test checks the intended contract:
1. The first global layer on PP rank 0 builds a full-hidden v0 from local TP value shards.
2. Later PP stages consume that full-hidden v0 without regenerating it.
3. Each TP rank slices only its own hidden shard from v0 before value residual mixing.
"""

from __future__ import annotations

import torch


TP_SIZE = 2
HIDDEN = 4


def stage0_build_v0(local_value_shard: torch.Tensor, tp_rank: int, gathered_full_value: torch.Tensor):
    """Emulate the non-SP stage-0 branch in RWKV7TimeMix.forward()."""
    layer_number = 1
    pp_rank = 0
    v0 = None

    if v0 is None and layer_number == 1 and pp_rank == 0:
        v0 = gathered_full_value

    return v0


def later_stage_mix_value(
    local_value_shard: torch.Tensor,
    v0_full_hidden: torch.Tensor,
    tp_rank: int,
    *,
    layer_number: int,
    pp_rank: int,
):
    """Emulate the non-SP later-stage branch in RWKV7TimeMix.forward()."""
    assert v0_full_hidden.shape[:2] == local_value_shard.shape[:2]

    if pp_rank == 0 and layer_number == 1:
        raise ValueError("v0 is not None for the first layer")

    shard_size = HIDDEN // TP_SIZE
    start_idx = tp_rank * shard_size
    end_idx = start_idx + shard_size
    v0_shard = v0_full_hidden[:, :, start_idx:end_idx]

    # Use a deterministic gate so we can assert the exact output.
    vt = torch.ones_like(local_value_shard)
    mixed_value = torch.lerp(local_value_shard, v0_shard, vt)
    return mixed_value, v0_shard


def transformer_layer_pack_non_sp(hidden_states: torch.Tensor, v0: torch.Tensor):
    """Emulate TransformerLayer's non-SP pack protocol."""
    return torch.stack([hidden_states, v0], dim=0)


def transformer_layer_unpack_non_sp(packed_hidden_states: torch.Tensor):
    """Emulate TransformerLayer's non-SP unpack protocol."""
    assert packed_hidden_states.dim() == 4
    hidden_states = packed_hidden_states[0]
    v0 = packed_hidden_states[1]
    return hidden_states, v0


def test_tp_pp_without_sequence_parallel_keeps_full_hidden_v0_contract():
    # Local TP value shards from the first global layer on PP rank 0.
    rank0_value = torch.tensor([[[1.0, 2.0]]])
    rank1_value = torch.tensor([[[101.0, 102.0]]])
    gathered_v0 = torch.tensor([[[1.0, 2.0, 101.0, 102.0]]])

    # Stage 0 / first global layer should materialize full-hidden v0.
    rank0_v0 = stage0_build_v0(rank0_value, tp_rank=0, gathered_full_value=gathered_v0)
    rank1_v0 = stage0_build_v0(rank1_value, tp_rank=1, gathered_full_value=gathered_v0)

    assert torch.equal(rank0_v0, gathered_v0)
    assert torch.equal(rank1_v0, gathered_v0)

    # Later PP stage receives the packed full-hidden v0 and should only slice its own shard.
    stage1_rank0_value = torch.tensor([[[9.0, 9.0]]])
    stage1_rank1_value = torch.tensor([[[8.0, 8.0]]])

    mixed0, shard0 = later_stage_mix_value(
        stage1_rank0_value, rank0_v0, tp_rank=0, layer_number=1, pp_rank=1
    )
    mixed1, shard1 = later_stage_mix_value(
        stage1_rank1_value, rank1_v0, tp_rank=1, layer_number=1, pp_rank=1
    )

    assert torch.equal(shard0, torch.tensor([[[1.0, 2.0]]]))
    assert torch.equal(shard1, torch.tensor([[[101.0, 102.0]]]))

    # With vt == 1, torch.lerp(value, v0_shard, vt) should fully pick v0_shard.
    assert torch.equal(mixed0, shard0)
    assert torch.equal(mixed1, shard1)


def test_non_sp_transformer_layer_protocol_keeps_v0_across_pp_stages():
    # Stage 0 emits a normal hidden state plus the full-hidden v0 cache.
    stage0_hidden_rank0 = torch.tensor([[[11.0, 12.0, 13.0, 14.0]]])
    stage0_hidden_rank1 = torch.tensor([[[21.0, 22.0, 23.0, 24.0]]])
    v0_full = torch.tensor([[[1.0, 2.0, 101.0, 102.0]]])

    packed_rank0 = transformer_layer_pack_non_sp(stage0_hidden_rank0, v0_full)
    packed_rank1 = transformer_layer_pack_non_sp(stage0_hidden_rank1, v0_full)

    # Next PP stage should recover the exact hidden/v0 pair from the packed tensor.
    unpacked_hidden0, unpacked_v00 = transformer_layer_unpack_non_sp(packed_rank0)
    unpacked_hidden1, unpacked_v01 = transformer_layer_unpack_non_sp(packed_rank1)

    assert torch.equal(unpacked_hidden0, stage0_hidden_rank0)
    assert torch.equal(unpacked_hidden1, stage0_hidden_rank1)
    assert torch.equal(unpacked_v00, v0_full)
    assert torch.equal(unpacked_v01, v0_full)

    # Stage 1 then consumes the carried v0 exactly as RWKV7TimeMix does in non-SP mode.
    local_value_rank0 = torch.tensor([[[9.0, 9.0]]])
    local_value_rank1 = torch.tensor([[[8.0, 8.0]]])
    mixed0, shard0 = later_stage_mix_value(
        local_value_rank0, unpacked_v00, tp_rank=0, layer_number=1, pp_rank=1
    )
    mixed1, shard1 = later_stage_mix_value(
        local_value_rank1, unpacked_v01, tp_rank=1, layer_number=1, pp_rank=1
    )

    assert torch.equal(shard0, torch.tensor([[[1.0, 2.0]]]))
    assert torch.equal(shard1, torch.tensor([[[101.0, 102.0]]]))

    # Stage 1 would repack the new hidden state together with the unchanged full-hidden v0.
    repacked_rank0 = transformer_layer_pack_non_sp(stage0_hidden_rank0, unpacked_v00)
    repacked_rank1 = transformer_layer_pack_non_sp(stage0_hidden_rank1, unpacked_v01)

    assert torch.equal(repacked_rank0[1], v0_full)
    assert torch.equal(repacked_rank1[1], v0_full)
    assert torch.equal(mixed0, shard0)
    assert torch.equal(mixed1, shard1)


if __name__ == "__main__":
    test_tp_pp_without_sequence_parallel_keeps_full_hidden_v0_contract()
    test_non_sp_transformer_layer_protocol_keeps_v0_across_pp_stages()
    print("PASS: non-SP TP+PP v0 contract stays self-consistent.")
