#!/usr/bin/env python3
"""Build a Qwen-vocabulary RWKV7 small model by deterministic tensor slicing.

This is an initialization model for smoke tests and distillation.  It keeps the
source vocabulary rows intact while taking leading hidden/layer slices from the
28B mapped checkpoint.  The resulting model must be trained; slicing does not
preserve the 28B model's function exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch


def slice_checkpoint(source: str, output: str, *, layers: int, hidden: int, heads: int,
                     ffn: int, decay_rank: int, learn_rank: int, gate_rank: int,
                     value_rank: int) -> int:
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {output}")
    if min(layers, hidden, heads, ffn, decay_rank, learn_rank, gate_rank, value_rank) <= 0:
        raise ValueError("target dimensions must be positive")
    if hidden % heads:
        raise ValueError("hidden must be divisible by heads")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "emb.weight" not in state:
        raise ValueError("expected a plain RWKV-LM state dict")
    source_hidden = state["emb.weight"].shape[1]
    source_layers = max(int(m.group(1)) for k in state for m in [re.match(r"blocks\.(\d+)\.", k)] if m) + 1
    if not (layers <= source_layers and hidden <= source_hidden):
        raise ValueError("target dimensions exceed source checkpoint")
    source_heads, head_size = state["blocks.0.att.r_k"].shape
    if heads > source_heads or hidden // heads != head_size:
        raise ValueError("slicing must retain the source head size")
    if ffn > state["blocks.0.ffn.key.weight"].shape[0]:
        raise ValueError("target FFN exceeds source checkpoint")
    for name, rank in (("w1", decay_rank), ("a1", learn_rank), ("g1", gate_rank), ("v1", value_rank)):
        if rank > state[f"blocks.0.att.{name}"].shape[1]:
            raise ValueError(f"target {name} rank exceeds source checkpoint")

    rank_by_name = {"w1": decay_rank, "w2": decay_rank, "a1": learn_rank,
                    "a2": learn_rank, "v1": value_rank, "v2": value_rank,
                    "g1": gate_rank, "g2": gate_rank}
    result = {}
    for key, tensor in state.items():
        if key.startswith("blocks."):
            match = re.match(r"blocks\.(\d+)\.(.*)", key)
            assert match is not None
            layer = int(match.group(1))
            if layer >= layers:
                continue
            suffix = match.group(2)
            if suffix.endswith(".r_k"):
                value = tensor[:heads, : hidden // heads].contiguous()
            elif suffix.rsplit(".", 1)[-1] in rank_by_name:
                name = suffix.rsplit(".", 1)[-1]
                rank = rank_by_name[name]
                if tensor.ndim == 2 and tensor.shape[0] == source_hidden:
                    value = tensor[:hidden, :rank].contiguous()
                elif tensor.ndim == 2 and tensor.shape[1] == source_hidden:
                    value = tensor[:rank, :hidden].contiguous()
                else:
                    value = tensor.clone()
            elif tensor.ndim == 3 and tensor.shape[-1] == source_hidden:
                value = tensor[..., :hidden].contiguous()
            elif tensor.ndim == 2 and tensor.shape == (source_hidden, source_hidden):
                value = tensor[:hidden, :hidden].contiguous()
            elif tensor.ndim == 2 and tensor.shape[1] == source_hidden:
                value = tensor[:ffn, :hidden].contiguous()
            elif tensor.ndim == 2 and tensor.shape[0] == source_hidden:
                value = tensor[:hidden, :ffn].contiguous()
            elif tensor.ndim == 1 and tensor.shape[0] == source_hidden:
                value = tensor[:hidden].contiguous()
            else:
                value = tensor.clone()
            result[key] = value
        elif key in ("emb.weight", "head.weight"):
            result[key] = tensor[:, :hidden].contiguous()
        elif key.startswith("ln_out."):
            result[key] = tensor[:hidden].contiguous()

    expected_blocks = layers
    actual_blocks = len({int(m.group(1)) for k in result for m in [re.match(r"blocks\.(\d+)\.", k)] if m})
    if actual_blocks != expected_blocks:
        raise ValueError(f"expected {expected_blocks} blocks, got {actual_blocks}")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.save(result, output)
    count = sum(t.numel() for t in result.values())
    with open(output + ".json", "w") as handle:
        json.dump({"source": source, "output": output, "parameters": count,
                   "layers": layers, "hidden": hidden, "heads": heads, "head_size": head_size,
                   "ffn": ffn, "vocab": result["emb.weight"].shape[0],
                   "ranks": {"decay": decay_rank, "learn": learn_rank, "gate": gate_rank, "value": value_rank},
                   "method": "leading tensor slices and first N layers; all vocabulary rows retained",
                   "quality": "untrained small initialization; not function-preserving compression"}, handle, indent=2)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=2304)
    parser.add_argument("--heads", type=int, default=36)
    parser.add_argument("--ffn", type=int, default=9216)
    parser.add_argument("--decay-rank", type=int, default=128)
    parser.add_argument("--learn-rank", type=int, default=128)
    parser.add_argument("--gate-rank", type=int, default=288)
    parser.add_argument("--value-rank", type=int, default=64)
    args = parser.parse_args()
    if args.hidden % args.heads:
        raise ValueError("hidden must be divisible by heads")
    count = slice_checkpoint(args.source, args.output, layers=args.layers, hidden=args.hidden,
                             heads=args.heads, ffn=args.ffn, decay_rank=args.decay_rank,
                             learn_rank=args.learn_rank, gate_rank=args.gate_rank,
                             value_rank=args.value_rank)
    print(f"saved {args.output}: {count:,} parameters ({count / 1e9:.3f}B)")


if __name__ == "__main__":
    main()
