"""
Regression test for RWKV7 HuggingFace LoRA loader mapping.

Expected workflow:
1. Run this script before fixing rwkv7_model.py. It should fail because the
   HuggingFace loader wires lora.0 / lora.2 into weight_up / weight_down backwards.
2. After fixing the mapping, this script should pass and confirm the imported
   bottleneck function matches the original HuggingFace two-linear LoRA function.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


HF_MODEL_DIR = Path(r"D:\fsdownload\RWKV7-Goose-World2.9-0.4B-HF")


def _hf_lora_reference(x: torch.Tensor, weight_0: torch.Tensor, weight_2: torch.Tensor) -> torch.Tensor:
    hidden = torch.nn.functional.linear(x, weight_0)
    return torch.nn.functional.linear(hidden, weight_2)


def _megatron_bottleneck(x: torch.Tensor, weight_down: torch.Tensor, weight_up: torch.Tensor) -> torch.Tensor:
    neck = torch.einsum("...d,dr->...r", x, weight_down)
    return torch.einsum("...r,rd->...d", neck, weight_up)


def _assert_close(name: str, got: torch.Tensor, expected: torch.Tensor):
    assert torch.allclose(got, expected, atol=5e-4, rtol=1e-5), (
        f"{name} mismatch\n"
        f"max_abs={torch.max(torch.abs(got - expected)).item()}\n"
    )


def test_hf_lora_weights_map_to_megatron_bottleneck():
    state_dict = load_file(str(HF_MODEL_DIR / "model.safetensors"))
    cases = [
        "model.layers.0.attn.a_lora.lora",
        "model.layers.0.attn.w_lora.lora",
        "model.layers.0.attn.g_lora.lora",
        "model.layers.1.attn.v_lora.lora",
    ]

    generator = torch.Generator().manual_seed(1234)

    for prefix in cases:
        weight_0 = state_dict[prefix + ".0.weight"].float()
        weight_2 = state_dict[prefix + ".2.weight"].float()
        x = torch.randn(3, weight_0.shape[1], generator=generator)

        reference = _hf_lora_reference(x, weight_0, weight_2)

        current_loader_result = _megatron_bottleneck(
            x,
            weight_down=weight_2,
            weight_up=weight_0,
        )
        assert not torch.allclose(current_loader_result, reference, atol=5e-4, rtol=1e-5), (
            f"current loader unexpectedly matched reference for {prefix}"
        )

        fixed_loader_result = _megatron_bottleneck(
            x,
            weight_down=weight_0.transpose(0, 1).contiguous(),
            weight_up=weight_2.transpose(0, 1).contiguous(),
        )
        _assert_close(prefix, fixed_loader_result, reference)


if __name__ == "__main__":
    test_hf_lora_weights_map_to_megatron_bottleneck()
    print("PASS: RWKV7 HuggingFace LoRA loader mapping matches the original two-linear function.")
