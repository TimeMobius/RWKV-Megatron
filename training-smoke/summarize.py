#!/usr/bin/env python3
"""Summarize comparable RWKV 3B training logs."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import yaml


TRAIN = re.compile(
    r"\] iteration\s+(?P<step>\d+)/\s*(?P<total>\d+)"
    r" \| consumed samples:\s*(?P<samples>\d+)"
    r" \| elapsed time per iteration \(ms\):\s*(?P<ms>[\d.]+)"
    r".*?\| lm loss:\s*(?P<loss>[\d.Ee+-]+)"
    r".*?\| number of skipped iterations:\s*(?P<skipped>\d+)"
    r" \| number of nan iterations:\s*(?P<nan>\d+)"
)
VALIDATION = re.compile(
    r"validation loss at iteration\s+(?P<step>\d+)"
    r"(?P<final> on validation set)? \| lm loss value:\s*(?P<loss>[\d.Ee+-]+)"
)


def summarize(path, config, warmup_steps):
    iterations = []
    validation = {}
    completed = False
    for line in path.read_text(errors="replace").splitlines():
        match = TRAIN.search(line)
        if match:
            row = match.groupdict()
            iterations.append(
                {
                    "step": int(row["step"]),
                    "samples": int(row["samples"]),
                    "ms": float(row["ms"]),
                    "loss": float(row["loss"]),
                    "skipped": int(row["skipped"]),
                    "nan": int(row["nan"]),
                }
            )
        match = VALIDATION.search(line)
        if match:
            key = match["step"] + ("-final" if match["final"] else "")
            validation[key] = float(match["loss"])
        if "[after training is done]" in line:
            completed = True

    expected = list(range(1, config["train_iters"] + 1))
    observed = [row["step"] for row in iterations]
    if not completed or observed != expected:
        raise ValueError(f"{path}: incomplete run; observed steps {observed}, expected {expected}")
    warm = [row["ms"] for row in iterations if row["step"] > warmup_steps]
    if not warm:
        raise ValueError("warmup_steps leaves no measured training iterations")
    tokens_per_step = config["global_batch_size"] * config["seq_length"]
    return {
        "log": str(path),
        "steps": len(iterations),
        "first_step_ms": iterations[0]["ms"],
        "warmup_excluded_steps": warmup_steps,
        "median_step_ms": statistics.median(warm),
        "mean_step_ms": statistics.mean(warm),
        "p95_step_ms": sorted(warm)[math.ceil(0.95 * len(warm)) - 1],
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": len(warm) * tokens_per_step / (sum(warm) / 1000),
        "tokens_per_second_per_gpu": len(warm) * tokens_per_step / (sum(warm) / 1000) / 8,
        "loss_by_step": {str(row["step"]): row["loss"] for row in iterations},
        "validation_loss": validation,
        "skipped_iterations": max(row["skipped"] for row in iterations),
        "nan_iterations": max(row["nan"] for row in iterations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config-3b.yaml"))
    parser.add_argument("--warmup-steps", type=int, default=5)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    runs = [summarize(path, config, args.warmup_steps) for path in args.logs]
    result = {"runs": runs}
    if len(runs) > 1:
        baseline = runs[0]
        comparisons = []
        for run in runs[1:]:
            shared = set(baseline["loss_by_step"]) & set(run["loss_by_step"])
            shared_validation = set(baseline["validation_loss"]) & set(run["validation_loss"])
            comparisons.append(
                {
                    "log": run["log"],
                    "speedup": run["tokens_per_second"] / baseline["tokens_per_second"],
                    "max_train_loss_delta": max(
                        abs(run["loss_by_step"][step] - baseline["loss_by_step"][step])
                        for step in shared
                    ),
                    "validation_loss_delta": {
                        key: run["validation_loss"][key] - baseline["validation_loss"][key]
                        for key in sorted(shared_validation)
                    },
                }
            )
        result["comparisons_to_first_run"] = comparisons
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
