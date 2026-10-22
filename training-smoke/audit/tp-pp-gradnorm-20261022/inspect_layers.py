"""Locate gradient concentrations in captured full-sequence, head-sharded tensors."""
import argparse
import json
import re
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(8)


def load(folder, layer, label, mb, gradient=False):
    suffix = "-grad.pt" if gradient else ".pt"
    files = list(folder.glob(f"layer{layer}-{label}-mb{mb}-rank*{suffix}"))
    pattern = rf"layer{layer}-{label}-mb{mb}-rank\d+" + (r"-grad\.pt" if gradient else r"\.pt")
    files = [path for path in files if re.fullmatch(pattern, path.name)]
    files.sort(key=lambda path: int(re.search(r"rank(\d+)", path.name)[1]))
    assert files, (folder, layer, label, mb)
    values = [torch.load(path, mmap=True, weights_only=True) for path in files]
    return torch.cat([value if gradient else value["output"] for value in values], dim=-1)


def metrics(a, b):
    a, b = a.float(), b.float()
    return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                actual_norm=float(a.norm()), reference_norm=float(b.norm()),
                max_abs=float((a-b).abs().max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--reference")
    parser.add_argument("--layers", default="0,3,15,19,23")
    parser.add_argument("--output", default="layer-inspection.json")
    args = parser.parse_args()
    result = {}
    for layer in map(int, args.layers.split(",")):
        for label in ["key", "receptance", "value", "wkv"]:
            for mb in range(2):
                output = load(HERE/args.folder, layer, label, mb)
                grad = load(HERE/args.folder, layer, label, mb, True)
                heads = grad.float().reshape(grad.shape[0], -1, 80, 64)
                squares = heads.square().sum((0, 1, 3))
                worst = squares.topk(5).indices.tolist()
                row = dict(output_norm=float(output.float().norm()), gradient_norm=float(grad.float().norm()),
                           top_heads=[dict(head=h, gradient_norm=float(squares[h].sqrt()),
                                           max_token=int(heads[:, :, h].square().sum((1, 2)).argmax())) for h in worst])
                if args.reference:
                    row["output_comparison"] = metrics(output, load(HERE/args.reference, layer, label, mb))
                    row["gradient_comparison"] = metrics(grad, load(HERE/args.reference, layer, label, mb, True))
                result[f"layer{layer}-{label}-mb{mb}"] = row
    (HERE/args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
