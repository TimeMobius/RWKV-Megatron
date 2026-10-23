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
        historical = "original" in case
        tag = "dp2-tp2-sp-pp2-bf16" if "-sp-" in case else "dp2-tp2-pp2-bf16"
        baseline_out = (
            ROOT / "training-smoke/dp2-distill-20261023" / tag
            if historical
            else OUT / case
        )
        baseline_ckpt = (
            ROOT / "training-smoke/checkpoints/dp2-distill-20261023" / tag
            if historical
            else CKPTS / case
        )
        resumed_out = OUT / (case if historical else case + "-resume")
        resumed_ckpt = CKPTS / (case if historical else case + "-resume")
        info["comparison_type"] = "historical_step3" if historical else "fresh_pair"
        for rank in range(8):
            a = torch.load(baseline_out / f"step3-rank{rank}.pt", weights_only=False)
            b = torch.load(resumed_out / f"step3-rank{rank}.pt", weights_only=False)
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
            for field in ("skipped", "grad_norm"):
                if a[field] != b[field]:
                    diff.append(dict(field=field, reference=a[field], actual=b[field]))
            for field in (
                "lm loss",
                "distill/kl_loss",
                "distill/student_ce_loss",
                "distill/ce_loss",
            ):
                if a["losses"].get(field) != b["losses"].get(field):
                    diff.append(
                        dict(
                            field=field,
                            reference=a["losses"].get(field),
                            actual=b["losses"].get(field),
                        )
                    )
            info["ranks"][str(rank)] = diff
        paths = sorted((baseline_ckpt / "iter_0000003").glob("*/model_optim_rng.pt"))
        assert len(paths) == 4, (case, "missing checkpoint shards", len(paths))
        counts = dict(tensors=0, elements=0)
        for path in paths:
            other = resumed_ckpt / "iter_0000003" / path.parent.name / path.name
            a = torch.load(path, map_location="cpu", weights_only=False)
            b = torch.load(other, map_location="cpu", weights_only=False)
            for field in (
                "model",
                "optimizer",
                "rng_state",
                "opt_param_scheduler",
                "iteration",
            ):
                try:
                    exact(
                        a[field],
                        b[field],
                        path.parent.name + "/" + field,
                        counts,
                    )
                except AssertionError as error:
                    info["checkpoint_differences"].append(str(error))
            assert (
                a["args"].consumed_train_samples
                == b["args"].consumed_train_samples
                == 48
            )
        kernel_diff = []
        kernel_counts = {}
        for rank in range(8):
            a = baseline_out / f"kernels-step3-rank{rank}.json"
            b = resumed_out / f"kernels-step3-rank{rank}.json"
            if a.exists() and b.exists():
                x, y = json.loads(a.read_text()), json.loads(b.read_text())
                assert x, "No autotuner configs were captured"
                kernel_counts[str(rank)] = [len(x), len(y)]
                kernel_diff.extend(
                    [f"{rank}/{k}" for k in set(x) | set(y) if x.get(k) != y.get(k)]
                )
        info["kernel_config_differences"] = kernel_diff
        info["kernel_config_compared_ranks"] = kernel_counts
        info["checkpoint_ranks"] = len(paths)
        info["checkpoint_exact_counts_before_first_failure_per_field"] = counts
        info["bit_identical"] = (
            not any(info["ranks"].values()) and not info["checkpoint_differences"]
        )
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
