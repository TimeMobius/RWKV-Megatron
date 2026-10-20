"""Summarize completed, identical-checkpoint sync/async tests; optionally publish evidence."""

import argparse
import ast
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics

import yaml


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]


def parse_run(entry):
    text = Path(entry["log"]).read_text()
    pattern = (r"iteration\s+(\d+)/\s*\d+.*elapsed time per iteration \(ms\): ([\d.]+)"
               r".*lm loss: ([\d.E+\-]+).*grad norm: ([\d.E+\-]+)"
               r".*number of skipped iterations:\s+(\d+).*number of nan iterations:\s+(\d+)")
    steps = [dict(iteration=int(i), ms=float(ms), loss=float(loss), grad_norm=float(grad),
                  skipped=int(skip), nan=int(nan))
             for i, ms, loss, grad, skip, nan in re.findall(pattern, text)]
    assert [x["iteration"] for x in steps] == list(range(198, 214))
    assert all(x["skipped"] == x["nan"] == 0 and math.isfinite(x["loss"]) and math.isfinite(x["grad_norm"])
               for x in steps)
    start = re.search(r"\[before the start of training step\] datetime: ([^\n]+)", text)[1].strip()
    end = re.search(r"\[exiting program at iteration 213\] datetime: ([^\n]+)", text)[1].strip()
    elapsed = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    config = yaml.safe_load(Path(entry["config"]).read_text())
    assert config["save"] is None
    assert "saving checkpoint at iteration" not in text
    assert "successfully loaded checkpoint" in text
    assert not any(x in text for x in ("Traceback (most recent call last)", "ChildFailedError"))
    result = dict(entry, steps=steps, completed_steps=len(steps), median_step_ms=statistics.median(x["ms"] for x in steps),
                  mean_step_ms=statistics.mean(x["ms"] for x in steps), first_step_ms=steps[0]["ms"],
                  train_wall_seconds_with_validation=elapsed,
                  validation_loss=float(re.search(r"validation loss at iteration 200 \| lm loss value: ([\d.E+\-]+)", text)[1]),
                  wandb_url=re.search(r"http://[^\s]+/runs/[a-z0-9]+", text)[0],
                  peak_allocated_mib=max(float(x) for x in re.findall(r"max allocated: ([\d.]+)", text)),
                  save_enabled=False)
    stats = re.search(r"async Qwen teacher final stats: (.+)", text)
    if stats:
        result["prefetch"] = ast.literal_eval(stats[1])
        assert result["prefetch"] == dict(submitted=16, consumed=16, completed=16, failed=0,
                                         active=0, peak_active=16, buffered=0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    meta = json.loads((FOLDER / "latest-async-validation.json").read_text())
    assert meta["sync"]["exit_code"] == meta["async"]["exit_code"] == 0
    runs = {mode: parse_run(meta[mode]) for mode in ("sync", "async")}
    precision = json.loads((FOLDER / "async-precision-results.json").read_text())
    assert precision["passed"]
    report = dict(passed=True, tests_passed=24, initial_iteration=197, final_iteration=213,
                  same_checkpoint=meta["checkpoint"], training=runs, precision=precision,
                  train_wall_speedup=runs["sync"]["train_wall_seconds_with_validation"] /
                                     runs["async"]["train_wall_seconds_with_validation"],
                  limits=["Only TP=PP=CP=1, single dataloader and rerun_mode=disabled are supported by async prefetch.",
                          "Precision replay freezes one real teacher response; independent live HTTP calls can differ.",
                          "Full-model audit checks first/final LayerNorm weight gradients, not every parameter gradient.",
                          "Short-run wall time includes teacher warmup and validation, excludes model setup and process cleanup.",
                          "Short-run throughput does not establish sustained long-run throughput or convergence."])
    path = FOLDER / "async-validation-results.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    if args.upload:
        import wandb

        with wandb.init(project="rwkv-qwen-distill-2p7b", entity="mobius", job_type="validation",
                        name="async-prefetch-validation", dir=str(ROOT / "training-smoke/wandb"),
                        settings=wandb.Settings(base_url="http://dian.sea-group.org:9098"),
                        config=dict(teacher_workers=16, lookahead=32, top_k=100, seq_length=128,
                                    initial_iteration=197, final_iteration=213, tests_passed=24,
                                    save_enabled=False)) as run:
            run.log({"validation/passed": 1, "validation/tests_passed": 24,
                     "benchmark/sync_wall_seconds": runs["sync"]["train_wall_seconds_with_validation"],
                     "benchmark/async_wall_seconds": runs["async"]["train_wall_seconds_with_validation"],
                     "benchmark/train_wall_speedup": report["train_wall_speedup"],
                     "teacher/peak_concurrent_batches": 16,
                     "precision/max_loss_abs": max(x["loss_abs"] for x in precision["checks"]),
                     "precision/max_logit_abs": max(x["logits"]["max_abs"] for x in precision["checks"])})
            artifact = wandb.Artifact("qwenvocab-2p7b-async-prefetch-validation", type="validation")
            files = [path, FOLDER / "async-precision-results.json", FOLDER / "ASYNC.md",
                     FOLDER / "async_precision.py", Path(meta["sync"]["config"]), Path(meta["async"]["config"]),
                     ROOT / "rwkv_megatron/online_teacher.py", ROOT / "rwkv_megatron/arguments.py",
                     ROOT / "pretrain_rwkv7.py", ROOT / "tests/test_teacher_prefetch.py",
                     ROOT / "tests/test_online_teacher.py",
                     ROOT / "configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-async-single.yaml"]
            for file in files:
                artifact.add_file(str(file), name=str(file.relative_to(ROOT)))
            run.log_artifact(artifact).wait()
            report["validation_wandb_url"] = run.url
            report["validation_artifact"] = f"{run.entity}/{run.project}/{artifact.name}"
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "train_wall_speedup": report["train_wall_speedup"],
                      "training": {mode: {key: value for key, value in result.items()
                                          if key not in ("steps", "config", "log")}
                                   for mode, result in runs.items()},
                      "validation_wandb_url": report.get("validation_wandb_url")}, indent=2))


if __name__ == "__main__":
    main()
