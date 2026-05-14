"""
Local smoke test for RWKV7 HuggingFace LoRA tensor shapes.

This verifies the local HF checkpoint uses the expected Linear(hidden -> rank)
then Linear(rank -> hidden) layout that the Megatron loader must transpose into
RWKV7Bottleneck's [hidden, rank] / [rank, hidden] storage.
"""

from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file


HF_MODEL_DIR = Path(r"D:\fsdownload\RWKV7-Goose-World2.9-0.4B-HF")


def test_local_hf_lora_shapes_match_expected_layout():
    state_dict = load_file(str(HF_MODEL_DIR / "model.safetensors"))
    expected = {
        "model.layers.0.attn.a_lora.lora.0.weight": (64, 1024),
        "model.layers.0.attn.a_lora.lora.2.weight": (1024, 64),
        "model.layers.0.attn.w_lora.lora.0.weight": (64, 1024),
        "model.layers.0.attn.w_lora.lora.2.weight": (1024, 64),
        "model.layers.0.attn.g_lora.lora.0.weight": (128, 1024),
        "model.layers.0.attn.g_lora.lora.2.weight": (1024, 128),
        "model.layers.1.attn.v_lora.lora.0.weight": (32, 1024),
        "model.layers.1.attn.v_lora.lora.2.weight": (1024, 32),
    }
    for key, shape in expected.items():
        actual = tuple(state_dict[key].shape)
        assert actual == shape, f"{key} shape mismatch: got {actual}, expected {shape}"


if __name__ == "__main__":
    test_local_hf_lora_shapes_match_expected_layout()
    print("PASS: local HF RWKV7 LoRA tensors use the expected hidden/rank layout.")
