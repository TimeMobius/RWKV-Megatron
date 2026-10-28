"""Real 2.7B/4096-token CP training using synthetic targets in the shared pool.

Exercises the production dataset, sample replacement, data staging, pipeline,
Adam, final checkpoint and validation. This is a local integration test, not a
Qwen quality measurement. Run with the usual CUDA/teacher tokenizer environment.
"""

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import uuid

import wandb
import yaml
import torch

from tools.run_distill_longrun import audit_training_log

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cases", nargs="+",
                        choices=["tp2-sp-pp2-cp1-dp2", "tp2-sp-pp2-cp2", "tp2-sp-cp4"],
                        default=["tp2-sp-pp2-cp1-dp2", "tp2-sp-pp2-cp2", "tp2-sp-cp4"])
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least two")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.cases:
        folder = output / name
        folder.mkdir(exist_ok=True)
        config_name = "tp2-sp-pp2-cp2" if name == "tp2-sp-pp2-cp1-dp2" else name
        cfg = yaml.safe_load((ROOT / "configs" / f"config-qwen-2p7b-distill-{config_name}-smoke.yaml").read_text())
        if name == "tp2-sp-pp2-cp1-dp2":
            cfg["model_parallel"].update(context_parallel_size=1, data_parallel_size=2)
        cfg.update(train_iters=args.steps, lr_decay_iters=args.steps, eval_interval=args.steps,
                   save=str(folder / "checkpoints"), tensorboard_dir=str(folder / "tensorboard"),
                   wandb_save_dir=str(folder / "wandb"),
                   wandb_exp_name=f"local-synthetic-2p7b-{name}-seq4096-{output.name}",
                   rwkv_teacher_sample_records=str(folder / "sample-selection"))
        path = folder / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        runid = uuid.uuid4().hex[:8]
        env = dict(os.environ, WANDB_MODE="online", WANDB_RUN_ID=runid,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   RWKV_CP_LOGITS_DIR=str(folder))
        peaks, observations = [0] * 8, []
        started = time.monotonic()
        with (folder / "training.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "tests/run_parallel_distill_frozen_smoke.py"), str(path)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                text = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=10)
                sample = [list(map(int, line.split(","))) for line in text.strip().splitlines()]
                for rank, memory, _ in sample:
                    peaks[rank] = max(peaks[rank], memory)
                observations.append(dict(seconds=time.monotonic() - started, gpus=sample))
                if time.monotonic() - started > 1200:
                    process.terminate()
                    process.wait(timeout=45)
                    raise TimeoutError(f"Local CP smoke exceeded 20 minutes: {folder}")
                time.sleep(0.5)
        (folder / "gpu-samples.json").write_text(json.dumps(observations) + "\n")
        item = dict(passed=False, exit_code=process.returncode, wall_seconds=time.monotonic() - started,
                    peak_gpu_memory_mib=peaks, real_student="2.7B", sequence_length=4096,
                    teacher="synthetic top100; full context asserted in pool",
                    wandb_id=runid, layout=cfg["model_parallel"])
        report[name] = item
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        assert process.returncode == 0, (name, process.returncode, folder / "training.log")
        item["audit"] = audit_training_log(folder / "training.log", 0)
        assert item["audit"]["last_iteration"] == args.steps, item["audit"]
        metrics = []
        for line in (folder / "training.log").read_text().splitlines():
            if re.search(r"iteration\s+\d+/\s*\d+.*consumed samples:", line):
                metrics.append({key: float(re.search(re.escape(key) + r":\s*(\S+)", line)[1])
                                for key in ("lm loss", "distill/kl_loss", "distill/student_ce_loss", "grad norm",
                                            "elapsed time per iteration (ms)")})
        item["step_metrics"] = metrics
        item["mean_step_seconds_excluding_first"] = sum(
            row["elapsed time per iteration (ms)"] / 1000 for row in metrics[1:]) / (len(metrics) - 1)
        remote = wandb.Api().run(f"mobius/rwkv-qwen-distill-2p7b/{runid}")
        assert remote.state == "finished", remote.state
        item.update(wandb=remote.url, wandb_state=remote.state,
                    validation_ce=remote.summary.get("distill/validation_student_ce_loss"),
                    validation_ppl=remote.summary.get("distill/validation_student_ppl"))
        tracker = folder / "checkpoints/latest_checkpointed_iteration.txt"
        assert int(tracker.read_text()) == args.steps
        item["final_checkpoint_iteration"] = args.steps
        parts = list((folder / "checkpoints" / f"iter_{args.steps:07d}").glob("mp_rank_*/model_optim_rng.pt"))
        expected_parts = (cfg["model_parallel"]["tensor_model_parallel_size"]
                          * cfg["model_parallel"]["pipeline_model_parallel_size"])
        assert len(parts) == expected_parts and all(part.stat().st_size for part in parts)
        item["final_checkpoint_parts"] = len(parts)
        logits = {}
        for snapshot in folder.glob("first-logits-rank*.pt"):
            rows = torch.load(snapshot, weights_only=True)
            assert not logits.keys() & rows.keys(), "DP readers must have disjoint samples"
            logits.update(rows)
        assert len(logits) == cfg["global_batch_size"], (name, len(logits))
        item["first_step_distinct_samples"] = len(logits)
        reference = report.get("tp2-sp-pp2-cp1-dp2")
        if reference is not None and name != "tp2-sp-pp2-cp1-dp2":
            expected = {}
            for snapshot in (output / "tp2-sp-pp2-cp1-dp2").glob("first-logits-rank*.pt"):
                expected.update(torch.load(snapshot, weights_only=True))
            assert logits.keys() == expected.keys(), "CP layouts must consume the same global batch"
            keys = sorted(expected)
            actual_logits = torch.stack([logits[key] for key in keys]).float()
            expected_logits = torch.stack([expected[key] for key in keys]).float()
            relative = ((actual_logits - expected_logits).norm() / expected_logits.norm()).item()
            assert torch.isfinite(actual_logits).all() and relative < 0.02, (name, relative)
            item["boundary_logits_relative_l2_vs_cp1"] = relative
            item["boundary_logits_bit_identical_vs_cp1"] = torch.equal(actual_logits, expected_logits)
            errors = {key: abs(metrics[0][key] - reference["step_metrics"][0][key])
                      / max(abs(reference["step_metrics"][0][key]), 1e-12)
                      for key in ("lm loss", "distill/kl_loss", "distill/student_ce_loss")}
            item["first_step_loss_relative_errors_vs_cp1"] = errors
            assert max(errors.values()) < 0.005, (name, errors)
        item["passed"] = True
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PASS LOCAL CP SMOKE", name, json.dumps(item), flush=True)


if __name__ == "__main__":
    main()
