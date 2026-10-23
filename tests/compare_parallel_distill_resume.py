"""Assert exact model, Adam state and FP32 master weights after resume."""

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training-smoke/checkpoints/parallel-distill-20261023"


def compare(a, b, path, counts):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), (path, (a.float() - b.float()).abs().max().item())
        counts["tensors"] += 1
        counts["elements"] += a.numel()
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            compare(a[key], b[key], f"{path}/{key}", counts)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for idx, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}/{idx}", counts)
    elif isinstance(a, (float, int, str, bool)) or a is None:
        assert a == b, (path, a, b)
    else:
        raise TypeError((path, type(a)))


if __name__ == "__main__":
    counts = dict(tensors=0, elements=0, ranks=0)
    for path in sorted((BASE / "combined/iter_0000003").glob("*/model_optim_rng.pt")):
        other = BASE / "combined-resume/iter_0000003" / path.parent.name / path.name
        a = torch.load(path, map_location="cpu", weights_only=False)
        b = torch.load(other, map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "opt_param_scheduler", "iteration"):
            compare(a[key], b[key], f"{path.parent.name}/{key}", counts)
        counts["ranks"] += 1
    assert counts["ranks"] == 4
    print(json.dumps(counts))
    (
        ROOT / "training-smoke/parallel-distill-20261023/resume-alignment.json"
    ).write_text(json.dumps(counts, indent=2) + "\n")
