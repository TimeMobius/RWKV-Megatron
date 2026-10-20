import pytest
import torch

from tools.build_rwkv7_small_from_checkpoint import slice_checkpoint


def test_small_checkpoint_keeps_vocabulary_and_head_layout(tmp_path):
    state = {
        "emb.weight": torch.randn(257, 128), "head.weight": torch.randn(257, 128),
        "ln_out.weight": torch.ones(128), "ln_out.bias": torch.zeros(128),
        "blocks.0.att.r_k": torch.randn(2, 64),
        "blocks.0.att.x_r": torch.randn(1, 1, 128),
        "blocks.0.ffn.key.weight": torch.randn(256, 128),
        "blocks.0.ffn.value.weight": torch.randn(128, 256),
        "blocks.1.att.r_k": torch.randn(2, 64),
    }
    for family in "wavg":
        state[f"blocks.0.att.{family}1"] = torch.randn(128, 16)
        state[f"blocks.0.att.{family}2"] = torch.randn(16, 128)
    src, dst = tmp_path / "source.pt", tmp_path / "small.pt"
    torch.save(state, src)
    dims = dict(layers=1, hidden=64, heads=1, ffn=128, decay_rank=8,
                learn_rank=8, gate_rank=8, value_rank=8)
    n = slice_checkpoint(str(src), str(dst), **dims)
    small = torch.load(dst, weights_only=True)
    assert n == sum(x.numel() for x in small.values())
    assert torch.equal(small["emb.weight"], state["emb.weight"][:, :64])
    assert torch.equal(small["head.weight"], state["head.weight"][:, :64])
    assert small["blocks.0.att.r_k"].shape == (1, 64)
    assert torch.equal(small["blocks.0.ffn.value.weight"], state["blocks.0.ffn.value.weight"][:64, :128])
    assert small["blocks.0.att.w1"].shape == (64, 8)
    assert small["blocks.0.att.w2"].shape == (8, 64)
    assert not any(k.startswith("blocks.1.") for k in small)
    assert (tmp_path / "small.pt.json").exists()
    with pytest.raises(FileExistsError):
        slice_checkpoint(str(src), str(dst), **dims)
    assert torch.equal(torch.load(src, weights_only=True)["emb.weight"], state["emb.weight"])
