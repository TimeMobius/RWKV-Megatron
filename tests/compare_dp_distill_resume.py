"""Strict DP2 resume assertions; record failures without relaxing to allclose."""

import json
import sys

import numpy as np
import torch
from prepare_dp_distill import CKPTS, LAYOUTS, OUT


def exact(a, b, path, counts):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
        counts["tensors"] += 1
        counts["elements"] += a.numel()
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            exact(a[key], b[key], f"{path}/{key}", counts)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for idx, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f"{path}/{idx}", counts)
    else:
        assert a == b, (path, a, b)


def check_case(tag, dp, tp, pp):
    counts = dict(tensors=0, elements=0, checkpoint_ranks=0, audit_ranks=0)
    paths = sorted((CKPTS / tag / "iter_0000003").glob("*/model_optim_rng.pt"))
    assert len(paths) == tp * pp, (tag, len(paths))
    for path in paths:
        other = CKPTS / f"{tag}-resumed/iter_0000003" / path.parent.name / path.name
        a = torch.load(path, map_location="cpu", weights_only=False)
        b = torch.load(other, map_location="cpu", weights_only=False)
        for key in (
            "model",
            "optimizer",
            "opt_param_scheduler",
            "iteration",
            "rng_state",
        ):
            exact(a[key], b[key], f"{tag}/{path.parent.name}/{key}", counts)
        assert (
            a["args"].consumed_train_samples == b["args"].consumed_train_samples == 48
        )
        counts["checkpoint_ranks"] += 1
    for rank in range(dp * tp * pp):
        a = torch.load(OUT / tag / f"step3-rank{rank}.pt", weights_only=False)
        b = torch.load(OUT / f"{tag}-resume/step3-rank{rank}.pt", weights_only=False)
        for key in (
            "samples",
            "grads",
            "before",
            "after",
            "losses",
            "logits",
            "skipped",
            "grad_norm",
        ):
            x, y = a[key], b[key]
            if key == "losses":
                # Request latency/memory legitimately change after restart.
                wanted = (
                    "lm loss",
                    "distill/kl_loss",
                    "distill/student_ce_loss",
                    "distill/ce_loss",
                )
                x = {name: x[name] for name in wanted} if x else {}
                y = {name: y[name] for name in wanted} if y else {}
            exact(x, y, f"{tag}/audit-rank{rank}/{key}", counts)
        counts["audit_ranks"] += 1
    return dict(**counts, bit_identical=True, sampler_and_rng_identical=True)


def main():
    cases = [
        (f"{name}-{dtype}", dp, tp, pp)
        for dtype in ("bf16", "fp32")
        for name, (dp, tp, pp, sp) in LAYOUTS.items()
        if dp == 2 and (dtype == "bf16" or name in ("dp2", "dp2-tp2-sp-pp2"))
    ]
    cases += [
        (name + "-deterministic", 2, 2, 2)
        for name in ("dp2-tp2-pp2-bf16", "dp2-tp2-sp-pp2-bf16")
    ]
    report = {}
    for tag, dp, tp, pp in cases:
        try:
            info = check_case(tag, dp, tp, pp)
            print("PASS DP RESUME", tag, info, flush=True)
        except AssertionError as error:
            info = dict(bit_identical=False, strict_assertion=str(error))
            print("FAIL STRICT DP RESUME", tag, str(error), flush=True)
        info["deterministic_mode"] = tag.endswith("-deterministic")
        report[tag] = info
    (OUT / "resume.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(any(not info["bit_identical"] for info in report.values()))


if __name__ == "__main__":
    sys.exit(main())
