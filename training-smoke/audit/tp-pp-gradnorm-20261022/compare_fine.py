"""Compare reconstructed sequence/head tensors at layer0..2 module boundaries."""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)
PATHS = ["input_layernorm", "self_attention.linear_receptance", "self_attention.linear_key",
         "self_attention.linear_value", "self_attention.core_attention", "self_attention.group_norm",
         "self_attention.bottleneck_gate", "self_attention.linear_proj", "self_attention",
         "pre_mlp_layernorm", "mlp.linear_up", "mlp.linear_down", "mlp", "layer"]


def metric(a, b):
    exact = torch.equal(a, b)
    af, bf = a.float(), b.float()
    return dict(bitwise_equal=exact, unequal_elements=int((a != b).sum()), numel=a.numel(),
                relative_l2=float((af - bf).norm() / bf.norm().clamp_min(1e-30)),
                max_abs=float((af - bf).abs().max()))


def first_tensor(value):
    while isinstance(value, (tuple, list)):
        value = value[0]
    return value


def load(folder, layer, name, tp, field):
    values = []
    for rank in range(tp):
        data = torch.load(HERE / folder / f"fine-layer{layer}-{name}-rank{rank}.pt", mmap=True, weights_only=True)
        tensor = first_tensor(data[field])
        if tensor.ndim == 4:
            tensor = tensor[0]  # Only the token-aligned hidden-state slot.
        values.append(tensor)
    if field == "output":
        local_heads = name in ["self_attention.linear_receptance", "self_attention.linear_key",
                               "self_attention.linear_value", "self_attention.core_attention",
                               "self_attention.group_norm", "self_attention.bottleneck_gate", "mlp.linear_up"]
    else:
        local_heads = name in ["self_attention.core_attention", "self_attention.group_norm",
                               "self_attention.linear_proj", "mlp.linear_down"]
    return torch.cat(values, dim=-1 if local_heads else 0)


if __name__ == "__main__":
    result = {}
    for layer in range(3):
        for name in PATHS:
            row = {}
            for field in ["inputs", "output"]:
                if name == "layer" and field == "inputs":
                    continue  # The block passes hidden_states by keyword.
                a = load("fine-tp2pp2", layer, name, 2, field)
                b = load("fine-tp4pp2", layer, name, 4, field)
                row[field] = metric(a, b)
                row[field]["token0"] = metric(a[:1], b[:1])
            result[f"layer{layer}-{name}"] = row
            print(f"layer{layer}-{name}", row, flush=True)
    (HERE / "fine-boundary-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
