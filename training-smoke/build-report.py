#!/usr/bin/env python3
"""Aggregate completed, repeated baseline/FLA/full-upstream benchmarks."""

import json
import statistics
from pathlib import Path

import yaml

from summarize import summarize


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "baseline": ["baseline-warm", "baseline-warm-repeat"],
    "fla_only": ["fla-only-warm", "fla-only-warm-repeat"],
    "full_upstream": ["megatron-upstream-30-final", "megatron-upstream-30-final-repeat"],
}


def main():
    config = yaml.safe_load((ROOT / "config-3b.yaml").read_text())
    exits = json.loads((ROOT / "validation-exits.json").read_text())
    result = {"date": "2026-10-09", "warmup_excluded_steps": 5, "groups": {}}
    for group, phases in GROUPS.items():
        runs = []
        for phase in phases:
            assert exits[phase]["exit_code"] == 0, exits[phase]
            run = summarize(ROOT / f"logs/{phase}.log", config, 5)
            assert run["nan_iterations"] == run["skipped_iterations"] == 0, run
            run["launcher_exit_code"] = 0
            runs.append(run)
        a, b = runs
        assert a["loss_by_step"] == b["loss_by_step"], group
        assert a["validation_loss"] == b["validation_loss"], group
        tps = [run["tokens_per_second"] for run in runs]
        result["groups"][group] = {
            "runs": runs,
            "mean_tokens_per_second": statistics.mean(tps),
            "mean_step_ms": statistics.mean(run["mean_step_ms"] for run in runs),
            "repeat_tps_spread_percent": (max(tps) - min(tps)) / statistics.mean(tps) * 100,
            "repeat_loss_exact_at_logged_precision": True,
        }
    baseline = result["groups"]["baseline"]
    for group in result["groups"].values():
        group["speedup_vs_baseline"] = group["mean_tokens_per_second"] / baseline["mean_tokens_per_second"]
        a, b = baseline["runs"][0], group["runs"][0]
        group["max_train_loss_delta_vs_baseline"] = max(
            abs(a["loss_by_step"][step] - b["loss_by_step"][step]) for step in a["loss_by_step"]
        )
        group["validation_loss_delta_vs_baseline"] = {
            key: b["validation_loss"][key] - a["validation_loss"][key]
            for key in a["validation_loss"]
        }
    result["resume"] = json.loads((ROOT / "results-resume.json").read_text())
    result["data_alignment"] = json.loads((ROOT / "results-data-alignment.json").read_text())
    result["limitations"] = [
        "Thirty steps do not establish convergence or downstream accuracy equivalence.",
        "Full update includes Python and TransformerEngine changes.",
        "Cold autotuning can change numerical trajectories; final results use repeated warm runs.",
        "Validated checkpoints use torch format and unchanged TP/PP topology.",
    ]
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, group in result["groups"].items():
        print(name, {key: group[key] for key in (
            "mean_tokens_per_second", "mean_step_ms", "speedup_vs_baseline",
            "repeat_tps_spread_percent", "max_train_loss_delta_vs_baseline",
        )})


if __name__ == "__main__":
    main()
