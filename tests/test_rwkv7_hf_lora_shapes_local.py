"""
Local smoke test for RWKV7 HuggingFace LoRA tensor shapes.

This verifies the local HF checkpoint uses the expected Linear(hidden -> rank)
then Linear(rank -> hidden) layout that the Megatron loader must transpose into
RWKV7Bottleneck's [hidden, rank] / [rank, hidden] storage.
"""

from __future__ import annotations

import os
from pathlib import Path

from safetensors.torch import load_file


def _resolve_hf_model_dir() -> Path:
    candidates = []
    env_path = os.environ.get("RWKV7_HF_MODEL_DIR")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/mnt/data/Models/Huggingface/RWKV7-Goose-World2.9-0.4B-HF"),
            Path(r"D:\fsdownload\RWKV7-Goose-World2.9-0.4B-HF"),
        ]
    )
    for candidate in candidates:
        if (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate RWKV7 HF model. Set RWKV7_HF_MODEL_DIR or place the model "
        "under /mnt/data/Models/Huggingface/RWKV7-Goose-World2.9-0.4B-HF."
    )


HF_MODEL_DIR = _resolve_hf_model_dir()


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
