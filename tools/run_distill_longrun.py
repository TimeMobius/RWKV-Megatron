"""Run two bounded online distillation phases with a checked checkpoint restart."""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def audit_training_log(path, expected_previous):
    text = path.read_text(errors="replace")
    iterations = []
    skipped = []
    nan = []
    teacher_failures = []
    for line in text.splitlines():
        match = re.search(r"iteration\s+(\d+)/\s*\d+.*consumed samples:", line)
        if match:
            if not math.isfinite(float(re.search(r"grad norm:\s*(\S+)", line)[1])):
                raise RuntimeError("Non-finite gradient norm")
            teacher_failures.append(
                float(re.search(r"teacher/failed:\s*(\S+)", line)[1])
            )
            iterations.append(int(match[1]))
            skipped.append(
                int(re.search(r"number of skipped iterations:\s*(\d+)", line)[1])
            )
            nan.append(int(re.search(r"number of nan iterations:\s*(\d+)", line)[1]))
    if not iterations or iterations[0] != expected_previous + 1:
        raise RuntimeError(
            f"Unexpected first iteration {iterations[:1]} after {expected_previous}"
        )
    if (
        iterations != list(range(iterations[0], iterations[-1] + 1))
        or any(skipped)
        or any(nan)
    ):
        raise RuntimeError("Training iterations are missing, skipped, or non-finite")
    if any(teacher_failures):
        raise RuntimeError("Teacher failures recorded in training log")
    if expected_previous and not re.search(
        rf"successfully loaded checkpoint.* at iteration {expected_previous}\b", text
    ):
        raise RuntimeError("Expected checkpoint load iteration missing from log")
    return dict(
        first_iteration=iterations[0],
        last_iteration=iterations[-1],
        optimizer_steps=len(iterations),
        skipped=sum(skipped),
        nan=sum(nan),
        teacher_failures=sum(teacher_failures),
    )


def checkpoint_iteration(folder):
    marker = folder / "latest_checkpointed_iteration.txt"
    iteration = int(marker.read_text().strip())
    if iteration < 1:
        raise RuntimeError("Expected a positive saved checkpoint iteration")
    checkpoint = folder / f"iter_{iteration:07d}"
    parts = list(checkpoint.glob("mp_rank_*/model_optim_rng.pt"))
    if len(parts) != 4 or any(path.stat().st_size == 0 for path in parts):
        raise RuntimeError(f"Incomplete TP2 PP2 checkpoint at {checkpoint}")
    return iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-minutes", type=int, default=360)
    parser.add_argument(
        "--smoke", action="store_true", help="Exit at step2, resume and finish step3"
    )
    args = parser.parse_args()
    if args.phase_minutes < 1:
        parser.error("phase-minutes must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    state_path = output / "status.json"
    state = dict(
        started=datetime.now().astimezone().isoformat(),
        smoke=args.smoke,
        supervisor_pid=os.getpid(),
        phases=[],
        state="preparing",
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )

    def save():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    process = None

    def stop(signum, frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise KeyboardInterrupt(f"signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    cfg = yaml.safe_load(
        (ROOT / "configs/config-qwen-2p7b-distill-tp2-sp-pp2-smoke.yaml").read_text()
    )
    cfg["model_parallel"].update(
        data_parallel_size=2,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        sequence_parallel=True,
    )
    cfg.update(
        deterministic_mode=True,
        seq_length=1024,
        rwkv_jsonl_chunk_chars=8192,
        global_batch_size=8,
        eval_global_batch_size=8,
        micro_batch_size=1,
        train_iters=3 if args.smoke else 1000000,
        lr_decay_iters=3 if args.smoke else 1000000,
        eval_iters=2,
        eval_interval=3 if args.smoke else 250,
        save=str(checkpoints),
        save_interval=2 if args.smoke else 250,
        rwkv_checkpoint_keep_last=2,
        load=None,
        exit_interval=2 if args.smoke else None,
        exit_duration_in_mins=None if args.smoke else args.phase_minutes,
        rwkv_teacher_pool=True,
        # This endpoint uses Qwen2Tokenizer's slow pretokenization regex.
        rwkv_teacher_tokenizer_use_fast=False,
        rwkv_teacher_replace_bad_samples=True,
        rwkv_teacher_sample_records=str(output / "sample-selection"),
        rwkv_teacher_async_prefetch=True,
        wandb_entity="mobius",
        wandb_project="rwkv-qwen-distill-2p7b",
        log_interval=1,
        num_workers=0,
    )
    # Real top-100 measurements favored eight concurrent single-prompt
    # requests over 16/24. Buffer four consumed global batches, independent
    # of DP/TP/PP readers; HTTP batching is unsupported by this endpoint.
    cfg.update(
        rwkv_teacher_pool_workers=8,
        rwkv_teacher_prefetch_workers=8,
        rwkv_teacher_prefetch_min_workers=4,
        rwkv_teacher_prefetch_latency_target=30.0,
        rwkv_teacher_prefetch_lookahead=4 * cfg["global_batch_size"],
        rwkv_teacher_pool_queue_size=8 * cfg["global_batch_size"],
        rwkv_teacher_pool_max_results=8 * cfg["global_batch_size"],
    )
    try:
        save()
        from tools.check_teacher_tokenizer import check_teacher_tokenizer

        state["teacher_tokenizer_preflight"] = check_teacher_tokenizer(cfg)
        save()
        for phase in (1, 2):
            if phase == 2:
                previous = checkpoint_iteration(checkpoints)
                cfg.update(load=str(checkpoints), exit_interval=None)
            else:
                previous = 0
            runid = uuid.uuid4().hex[:8]
            phase_name = f"phase{phase}"
            cfg.update(
                wandb_exp_name=f"{output.name}-{phase_name}",
                wandb_save_dir=str(output / phase_name / "wandb"),
                tensorboard_dir=str(output / phase_name / "tensorboard"),
            )
            path = output / (phase_name + ".yaml")
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            entry = dict(
                phase=phase,
                load_iteration=previous,
                wandb_id=runid,
                started=datetime.now().astimezone().isoformat(),
            )
            state["phases"].append(entry)
            state["state"] = "running"
            with (output / (phase_name + ".log")).open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "tools/launch_distillation.py"),
                        "--rwkv-config",
                        str(path),
                    ],
                    cwd=ROOT,
                    env=dict(
                        os.environ,
                        WANDB_RUN_ID=runid,
                        WANDB_MODE="online",
                        WANDB_TAGS="distill,2p7b,long-sequence,checkpoint-restart,"
                        + ("smoke" if args.smoke else "12hour"),
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                entry["launcher_pid"] = process.pid
                save()
                entry["exit_code"] = process.wait()
            if entry["exit_code"] != 0:
                raise RuntimeError(
                    f'Phase {phase} exited {entry["exit_code"]}; see its log'
                )
            entry["log_audit"] = audit_training_log(
                output / (phase_name + ".log"), previous
            )
            saved = checkpoint_iteration(checkpoints)
            if saved <= previous:
                raise RuntimeError("Checkpoint iteration did not advance")
            entry.update(
                saved_iteration=saved, finished=datetime.now().astimezone().isoformat()
            )
            import wandb

            remote = wandb.Api().run("mobius/rwkv-qwen-distill-2p7b/" + runid)
            entry.update(wandb_url=remote.url, wandb_state=remote.state)
            if remote.state != "finished":
                raise RuntimeError(f"Unexpected W&B state {remote.state}")
            save()
        state["state"] = "completed"
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}")
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        state["updated"] = datetime.now().astimezone().isoformat()
        save()
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
