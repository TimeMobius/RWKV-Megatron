"""Run bounded online distillation, audit it, and evaluate the final checkpoint."""

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
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def audit_training_log(path, expected_previous):
    text = path.read_text(errors="replace")
    iterations = []
    skipped = []
    nan = []
    teacher_failures = []
    sample_errors = []
    failure_metric_names = set()
    for line in text.splitlines():
        match = re.search(r"iteration\s+(\d+)/\s*\d+.*consumed samples:", line)
        if match:
            def metric(name, optional=False):
                found = re.search(re.escape(name) + r":\s*(\S+)", line)
                if found is None:
                    if optional:
                        return None
                    raise RuntimeError(f"Missing {name} metric at iteration {match[1]}")
                value = float(found[1])
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {name} metric")
                return value

            metric("grad norm")
            failure_key = ("teacher/pool_fatal_errors" if "teacher/pool_fatal_errors:" in line
                           else "teacher/failed")
            failure_metric_names.add(failure_key)
            failures = metric(failure_key)
            if failures < 0:
                raise RuntimeError("Negative teacher failure counter")
            teacher_failures.append(failures)
            sample_errors.append(metric("teacher/pool_sample_errors", optional=True) or 0.0)
            iterations.append(int(match[1]))
            skipped.append(metric("number of skipped iterations"))
            nan.append(metric("number of nan iterations"))
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
        teacher_failures=max(teacher_failures),
        teacher_sample_errors=max(sample_errors),
        teacher_failure_metrics=sorted(failure_metric_names),
    )


def checkpoint_iteration(folder, tp=2, pp=2):
    marker = folder / "latest_checkpointed_iteration.txt"
    iteration = int(marker.read_text().strip())
    if iteration < 1:
        raise RuntimeError("Expected a positive saved checkpoint iteration")
    checkpoint = folder / f"iter_{iteration:07d}"
    parts = list(checkpoint.glob("mp_rank_*/model_optim_rng.pt"))
    expected = {f'mp_rank_{rank:02d}' + (f'_{stage:03d}' if pp > 1 else '')
                for rank in range(tp) for stage in range(pp)}
    if ({path.parent.name for path in parts} != expected
            or any(path.stat().st_size == 0 for path in parts)):
        raise RuntimeError(f"Incomplete TP{tp} PP{pp} checkpoint at {checkpoint}")
    return iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--rwkv-config', type=Path,
                        help='Use a tested model and parallel layout instead of the default 2.7B layout')
    parser.add_argument('--local-teacher', action='store_true',
                        help='Use synthetic top100 targets for local stability testing; no online teacher quality claim')
    parser.add_argument('--resume-from', type=Path,
                        help='Resume model, optimizer, RNG and data cursor from a complete checkpoint root')
    parser.add_argument('--sample-records', type=Path,
                        help='Reuse the sample selection journal when resuming')
    parser.add_argument('--teacher-retry-window', type=float, default=300.0)
    parser.add_argument('--teacher-step-timeout', type=float, default=360.0)
    parser.add_argument('--teacher-http-timeout', type=float, default=600.0)
    parser.add_argument("--phase-minutes", type=int, default=360)
    parser.add_argument('--seq-length', type=int, default=1024)
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--global-batch-size', type=int, default=8)
    parser.add_argument('--teacher-pool-workers', type=int, default=8)
    parser.add_argument('--context-parallel-size', type=int, choices=(1, 2), default=1,
                        help='Use DP2 with CP1, or DP1 with CP2, keeping TP2 x PP2 on eight GPUs')
    parser.add_argument('--save-interval', type=int, default=250)
    parser.add_argument('--keep-checkpoints', type=int, default=2)
    parser.add_argument('--no-recompute', action='store_true',
                        help='Keep activations when measured GPU capacity permits')
    parser.add_argument('--continuous-minutes', type=int,
                        help='One continuous training phase, with no planned checkpoint restart')
    parser.add_argument(
        "--smoke", action="store_true", help="Exit at step2, resume and finish step3"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load((args.rwkv_config or
                          ROOT / 'configs/config-qwen-2p7b-distill-tp2-sp-pp2-smoke.yaml').read_text())
    if args.rwkv_config:
        from tools.launch_distillation import validate_parallel_settings
        dp, tp, pp, cp = validate_parallel_settings(cfg)
        if args.context_parallel_size != 1 and args.context_parallel_size != cp:
            parser.error('--context-parallel-size conflicts with the supplied config')
    else:
        dp, tp, pp, cp = 2 // args.context_parallel_size, 2, 2, args.context_parallel_size
    if (args.seq_length < 2 or args.micro_batch_size < 1 or args.teacher_pool_workers < 1
            or args.global_batch_size < 1 or args.save_interval < 1 or args.keep_checkpoints < 1
            or args.global_batch_size % (dp * args.micro_batch_size)
            or args.seq_length % (tp * cp)
            or (cp > 1 and args.seq_length % (64 * cp))):
        parser.error('sequence must divide TP x CP and 64 x CP with CP; global batch must divide DP times micro batch; save interval and keep-checkpoints must be positive')
    if args.phase_minutes < 1:
        parser.error("phase-minutes must be positive")
    if args.continuous_minutes is not None and (args.continuous_minutes < 1 or args.smoke):
        parser.error('continuous-minutes must be positive and cannot be combined with smoke')
    if args.resume_from and (args.smoke or args.continuous_minutes is None):
        parser.error('--resume-from requires continuous mode without --smoke')
    if (any(not math.isfinite(v) or v <= 0 for v in
            (args.teacher_retry_window, args.teacher_step_timeout, args.teacher_http_timeout))
            or args.teacher_step_timeout <= args.teacher_retry_window):
        parser.error('teacher timeouts must be positive and step timeout must exceed retry window')
    collective_timeout = (cfg.get('distributed_timeout_seconds_after_init')
                          or cfg.get('distributed_timeout_minutes', 30) * 60)
    if args.teacher_step_timeout + 60 >= collective_timeout:
        parser.error('distributed timeout must exceed teacher step timeout by over 60 seconds')
    resume_iteration = checkpoint_iteration(args.resume_from.resolve(), tp, pp) if args.resume_from else 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    state_path = output / "status.json"
    state = dict(
        started=now(),
        smoke=args.smoke,
        supervisor_pid=os.getpid(),
        phases=[],
        state="preparing",
        continuous_minutes=args.continuous_minutes,
        planned_restart=args.continuous_minutes is None,
        resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
        resume_iteration=resume_iteration,
        layout=dict(dp=dp, tp=tp, pp=pp, cp=cp,
                    sp=cfg.get('model_parallel', {}).get('sequence_parallel', True)),
        source_config=str(args.rwkv_config.resolve()) if args.rwkv_config else None,
        teacher_mode='local-synthetic' if args.local_teacher else 'online',
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
    cfg["model_parallel"].update(
        data_parallel_size=dp,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
    )
    if not args.rwkv_config:
        cfg['model_parallel']['sequence_parallel'] = True
    cfg.update(
        deterministic_mode=True,
        seq_length=args.seq_length,
        rwkv_jsonl_chunk_chars=max(8192, 8 * args.seq_length),
        global_batch_size=args.global_batch_size,
        eval_global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        eval_micro_batch_size=args.micro_batch_size,
        train_iters=3 if args.smoke else 1000000,
        lr_decay_iters=3 if args.smoke else 1000000,
        eval_iters=2,
        eval_interval=3 if args.smoke else 250,
        save=str(checkpoints),
        save_interval=2 if args.smoke else args.save_interval,
        rwkv_checkpoint_keep_last=args.keep_checkpoints,
        load=str(args.resume_from.resolve()) if args.resume_from else None,
        exit_interval=2 if args.smoke else None,
        exit_duration_in_mins=None if args.smoke else (args.continuous_minutes or args.phase_minutes),
        rwkv_teacher_pool=True,
        # This endpoint uses Qwen2Tokenizer's slow pretokenization regex.
        rwkv_teacher_tokenizer_use_fast=False,
        rwkv_teacher_replace_bad_samples=True,
        rwkv_teacher_sample_records=str(args.sample_records.resolve() if args.sample_records else output / "sample-selection"),
        rwkv_teacher_async_prefetch=True,
        rwkv_teacher_retry_window=args.teacher_retry_window,
        rwkv_teacher_step_timeout=args.teacher_step_timeout,
        rwkv_teacher_timeout=args.teacher_http_timeout,
        rwkv_teacher_save_on_failure=True,
        rwkv_teacher_retry_backoff_initial=5.0,
        rwkv_teacher_retry_backoff_max=30.0,
        wandb_entity="mobius",
        wandb_project=cfg.get('wandb_project') or 'rwkv-qwen-distill-2p7b',
        log_interval=1,
        num_workers=0,
    )
    if args.no_recompute:
        cfg['language_model'].update(recompute_granularity=None, recompute_method=None,
                                     recompute_num_layers=None)
    # Real top-100 measurements favored eight concurrent single-prompt
    # requests over 16/24. Buffer four consumed global batches, independent
    # of DP/TP/PP readers; HTTP batching is unsupported by this endpoint.
    cfg.update(
        rwkv_teacher_pool_workers=args.teacher_pool_workers,
        rwkv_teacher_pool_port=56329,
        rwkv_teacher_prefetch_workers=args.teacher_pool_workers,
        rwkv_teacher_prefetch_min_workers=max(1, args.teacher_pool_workers // 2),
        rwkv_teacher_prefetch_latency_target=30.0,
        rwkv_teacher_prefetch_lookahead=4 * cfg["global_batch_size"],
        rwkv_teacher_pool_queue_size=8 * cfg["global_batch_size"],
        rwkv_teacher_pool_max_results=8 * cfg["global_batch_size"],
    )
    if args.local_teacher:
        cfg['rwkv_teacher_model'] = 'local-synthetic-top100'
    try:
        save()
        from tools.check_teacher_tokenizer import check_teacher_tokenizer
        from tools.evaluate_distill_checkpoint import fixed_manifest
        from transformers import AutoTokenizer

        state["teacher_tokenizer_preflight"] = (
            dict(mode='local-synthetic', upstream_checked=False, optimizer_steps=0)
            if args.local_teacher else check_teacher_tokenizer(cfg))
        # Freeze evaluation tokens before training so data drift fails clearly
        # instead of silently changing the final checkpoint's validation set.
        tokenizer = AutoTokenizer.from_pretrained(
            cfg['rwkv_teacher_tokenizer'], trust_remote_code=True,
            local_files_only=True, use_fast=cfg['rwkv_teacher_tokenizer_use_fast'])
        manifest = fixed_manifest(output / 'fixed-validation.json', cfg, tokenizer)
        state['fixed_validation_sha256'] = manifest['sha256']
        del manifest, tokenizer
        save()
        for phase in ((1,) if args.continuous_minutes is not None else (1, 2)):
            if phase == 2:
                previous = checkpoint_iteration(checkpoints, tp, pp)
                cfg.update(load=str(checkpoints), exit_interval=None)
            else:
                previous = resume_iteration
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
                started=now(),
            )
            state["phases"].append(entry)
            state["state"] = "running"
            with (output / (phase_name + ".log")).open("w") as log:
                training_command = (
                    [sys.executable, str(ROOT / 'tests/run_parallel_distill_frozen_smoke.py'), str(path)]
                    if args.local_teacher else [
                        sys.executable,
                        str(ROOT / "tools/launch_distillation.py"),
                        "--rwkv-config",
                        str(path),
                    ])
                process = subprocess.Popen(
                    training_command,
                    cwd=ROOT,
                    env=dict(
                        os.environ,
                        WANDB_RUN_ID=runid,
                        WANDB_MODE="online",
                        WANDB_TAGS="distill,long-sequence,"
                        + ('synthetic-teacher,local-stability,' if args.local_teacher else 'online-teacher,')
                        + ('continuous,' if args.continuous_minutes is not None else 'checkpoint-restart,')
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
            saved = checkpoint_iteration(checkpoints, tp, pp)
            if saved <= previous:
                raise RuntimeError("Checkpoint iteration did not advance")
            entry.update(
                saved_iteration=saved, finished=now()
            )
            import wandb

            remote = wandb.Api().run(f"{cfg['wandb_entity']}/{cfg['wandb_project']}/{runid}")
            entry.update(wandb_url=remote.url, wandb_state=remote.state)
            if remote.state != "finished":
                raise RuntimeError(f"Unexpected W&B state {remote.state}")
            save()
        state['state'] = 'evaluating_checkpoint'
        save()
        quality_path = output / 'checkpoint-quality.json'
        with (output / 'checkpoint-quality.log').open('w') as log:
            process = subprocess.Popen(
                [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                 '--nproc-per-node=1', str(ROOT / 'tools/evaluate_distill_checkpoint.py'),
                 '--rwkv-config', str(path),
                 '--checkpoint', str(checkpoints / f'iter_{saved:07d}'),
                 '--manifest', str(output / 'fixed-validation.json'),
                 '--output', str(quality_path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            save()
            quality_exit = process.wait()
        if quality_exit:
            raise RuntimeError(f'Checkpoint quality evaluation exited {quality_exit}; see checkpoint-quality.log')
        from tools.evaluate_distill_checkpoint import upload_quality_report

        state['checkpoint_quality'] = upload_quality_report(quality_path, cfg)
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
        state["updated"] = now()
        save()
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
