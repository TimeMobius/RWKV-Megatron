"""Recover missing metrics from completed Megatron logs into the same W&B run.

Uses an explicit training-iteration axis because W&B history cannot insert old
internal steps. Existing training history is retained. No training is started.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

import yaml


BASE_URL = "http://dian.sea-group.org:9098"
PROJECT = "mobius/rwkv-qwen-distill-2p7b"
AXIS = "train/iteration"


def parse_metrics(text, config):
    seq_length = int(config["seq_length"])
    parallel = config.get("model_parallel", {})
    # These recovery logs are all from single-GPU runs. Do not infer a world
    # size from model-parallel degrees for a future data-parallel run.
    if any(parallel.get(key, 1) != 1 for key in (
        "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"
    )):
        raise ValueError("Log backfill currently requires a verified single-GPU run")
    rows, memory = [], {}
    pattern = (r"iteration\s+(\d+)/\s*\d+.*consumed samples:\s+(\d+)"
               r".*elapsed time per iteration \(ms\): ([\d.]+)"
               r".*throughput per GPU \(TFLOP/s/GPU\): ([\d.]+)"
               r".*global batch size:\s+(\d+).*lm loss: ([\d.E+\-]+)"
               r".*number of skipped iterations:\s+(\d+).*number of nan iterations:\s+(\d+)")
    memory_pattern = (r"\[Rank 0\] \((?:after (\d+) iterations|(?:before|after) save_checkpoint for iteration (\d+))\)"
                      r" memory \(MB\) \| allocated: ([\d.]+) \| max allocated: ([\d.]+)"
                      r" \| reserved: ([\d.]+) \| max reserved: ([\d.]+)")
    for match in re.finditer(memory_pattern, text):
        a, b, allocated, peak, reserved, peak_reserved = match.groups()
        memory[int(a or b)] = {
            "memory/allocated_mib": float(allocated), "memory/max_allocated_mib": float(peak),
            "memory/reserved_mib": float(reserved), "memory/max_reserved_mib": float(peak_reserved),
        }
    for match in re.finditer(pattern, text):
        iteration, consumed, ms, tflops, batch, loss, skipped, nan = match.groups()
        iteration, consumed, batch = int(iteration), int(consumed), int(batch)
        seconds = float(ms) / 1000
        if seconds <= 0 or not math.isfinite(seconds):
            raise ValueError("Invalid iteration time in source log")
        row = {AXIS: iteration, "iteration-time": seconds, "throughput": float(tflops),
               "throughput/tokens_per_sec": batch * seq_length / seconds,
               "throughput/samples_per_sec": batch / seconds,
               "train/consumed_samples": consumed, "train/consumed_input_tokens": consumed * seq_length,
               "train/skipped_iterations": int(skipped), "train/nan_iterations": int(nan)}
        if config.get("rwkv_distill_ce_weight", 0) == 0 and config.get("rwkv_distill_logit_weight", 1) > 0:
            row["distill/kl_loss"] = float(loss) / config.get("rwkv_distill_logit_weight", 1)
        row.update(memory.get(iteration, {}))
        rows.append(row)
    if not rows or any(a[AXIS] >= b[AXIS] for a, b in zip(rows, rows[1:])):
        raise ValueError("Expected nonempty, strictly increasing training iterations")
    final = re.search(r"async Qwen teacher final stats: (.+)", text)
    teacher = ast.literal_eval(final[1]) if final else None
    return rows, teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    text = args.log.read_text()
    if f"{BASE_URL}/{PROJECT}/runs/{args.run_id}" not in text:
        raise ValueError("Source log does not match the target W&B run")
    if not re.search(r"using world size:\s*1,", text):
        raise ValueError("Source log must confirm world size 1")
    config = yaml.safe_load(args.config.read_text())
    rows, teacher = parse_metrics(text, config)
    digest = hashlib.sha256(args.log.read_bytes()).hexdigest()
    report = dict(run_url=f"{BASE_URL}/{PROJECT}/runs/{args.run_id}", source_log=str(args.log.resolve()),
                  source_config=str(args.config.resolve()), source_sha256=digest, rows=rows,
                  teacher_final_snapshot=teacher, uploaded=False,
                  units={"iteration-time": "seconds", "throughput": "Megatron estimated TFLOP/s/GPU, rounded to 0.1 in source log",
                         "throughput/tokens_per_sec": "global batch * input sequence length / iteration seconds; excludes eval",
                         "memory/*_mib": "Sparse PyTorch allocator snapshots in MiB; only at logged iterations"},
                  unavailable=["per-step teacher request/wait latency and concurrency",
                               "CE and true perplexity", "measured RWKV hardware FLOPs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.upload:
        import wandb

        api = wandb.Api(overrides={"base_url": BASE_URL})
        existing = api.run(f"{PROJECT}/{args.run_id}")
        if existing.state != "finished":
            raise ValueError("Only backfill a finished run")
        if existing.summary.get("metrics_backfill/source_sha256") == digest:
            report.update(uploaded=True, already_uploaded=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"already_uploaded": True, "run_url": report["run_url"]}))
            return
        root = Path(__file__).resolve().parents[1]
        with wandb.init(project=PROJECT.split('/')[1], entity=PROJECT.split('/')[0],
                        id=args.run_id, resume="must", dir=str(root / "training-smoke/wandb"),
                        settings=wandb.Settings(base_url=BASE_URL, x_disable_stats=True, console="off")) as run:
            run.config.update({"metrics_backfill": dict(source_log=str(args.log.resolve()), source_sha256=digest,
                                                        step_axis=AXIS, original_start=str(existing.created_at),
                                                        origin="Recovered from original training stdout; no training resumed",
                                                        unavailable=report["unavailable"], units=report["units"])})
            run.define_metric(AXIS, summary="none")
            for key in sorted(set().union(*(set(x) for x in rows)) - {AXIS}):
                run.define_metric(key, step_metric=AXIS)
            for row in rows:
                run.log(row)
            interval_tokens = sum(config['seq_length'] * x['throughput/samples_per_sec'] * x['iteration-time'] for x in rows)
            summary = {"metrics_backfill/rows": len(rows), "metrics_backfill/source_sha256": digest,
                       "throughput/train_interval_tokens_per_sec": interval_tokens / sum(x['iteration-time'] for x in rows),
                       "metrics_backfill/completed": True}
            if teacher:
                summary.update({f"teacher/final_{key}": value for key, value in teacher.items()})
            for key in ("memory/max_allocated_mib", "memory/max_reserved_mib"):
                values = [x[key] for x in rows if key in x]
                if values:
                    summary[key] = max(values)
            run.summary.update(summary)
            report['uploaded'] = True
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            artifact = wandb.Artifact(f"distill-metrics-backfill-{args.run_id}", type="metrics-backfill")
            artifact.add_file(str(args.output))
            run.log_artifact(artifact).wait()
    print(json.dumps(dict(run_url=report['run_url'], rows=len(rows), uploaded=report['uploaded'],
                          metric_keys=sorted(set().union(*(set(x) for x in rows)))), indent=2))


if __name__ == "__main__":
    main()
