"""Record exactness and numeric drift across instrumented/control replay pairs."""

import argparse
import json
import sys
from pathlib import Path

import torch
from compare_dp_distill_resume import exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-numerics-20261023"
CKPTS = ROOT / "training-smoke/checkpoints/distill-numerics-20261023"
sys.path.insert(0, str(ROOT / "training-smoke/audit"))
from distributed import metric


def compare(cases):
    report = {}
    for case in cases:
        info = dict(ranks={}, checkpoint_differences=[])
        for rank in range(8):
            a = torch.load(OUT / case / f"step3-rank{rank}.pt", weights_only=False)
            b = torch.load(
                OUT / (case + "-resume") / f"step3-rank{rank}.pt", weights_only=False
            )
            diff = []
            assert a["samples"] == b["samples"]
            for field in ("grads", "before", "after", "logits"):
                for key, value in a[field].items():
                    if not torch.equal(value, b[field][key]):
                        diff.append(
                            dict(
                                field=field,
                                key=key,
                                metric=metric(b[field][key], value),
                            )
                        )
            info["ranks"][str(rank)] = diff
        for path in sorted(
            (CKPTS / case / "iter_0000003").glob("*/model_optim_rng.pt")
        ):
            other = (
                CKPTS
                / (case + "-resume")
                / "iter_0000003"
                / path.parent.name
                / path.name
            )
            a = torch.load(path, map_location="cpu", weights_only=False)
            b = torch.load(other, map_location="cpu", weights_only=False)
            for field in ("model", "optimizer", "rng_state"):
                try:
                    exact(
                        a[field],
                        b[field],
                        path.parent.name + "/" + field,
                        dict(tensors=0, elements=0),
                    )
                except AssertionError as error:
                    info["checkpoint_differences"].append(str(error))
        kernel_diff = []
        for rank in range(8):
            a = OUT / case / f"kernels-step3-rank{rank}.json"
            b = OUT / (case + "-resume") / f"kernels-step3-rank{rank}.json"
            if a.exists() and b.exists():
                x, y = json.loads(a.read_text()), json.loads(b.read_text())
                assert x, "No autotuner configs were captured"
                kernel_diff.extend(
                    [f"{rank}/{k}" for k in set(x) | set(y) if x.get(k) != y.get(k)]
                )
        info["kernel_config_differences"] = kernel_diff
        report[case] = info
        print(
            case,
            "snapshot diffs",
            sum(len(v) for v in info["ranks"].values()),
            "checkpoint",
            info["checkpoint_differences"],
            "kernel configs",
            len(kernel_diff),
        )
    path = OUT / "replay-controls.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(report)
    path.write_text(json.dumps(existing, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "combined",
            "combined-none",
            "combined-sync",
            "combined-single-thread",
        ],
    )
    compare(parser.parse_args().cases)
