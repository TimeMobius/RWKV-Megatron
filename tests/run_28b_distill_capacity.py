"""Compare 28B/4096-token distillation layouts on eight GPUs with local targets.

The student, data, shared pool, loss and Adam use the production path. Synthetic
top100 targets isolate capacity from the availability of the online teacher.
This probe does not save hundreds of GB of checkpoints or certify model quality.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

import wandb
import yaml

from tools.launch_distillation import default_training_command, launch
from tools.run_distill_longrun import audit_training_log

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "tp4-pp2": (4, 2, 1, 0.0),
    "tp2-pp4": (2, 4, 1, 0.0),
    "tp2-pp4-balanced": (2, 4, 1, 0.0),
    "tp2-pp4-balanced18": (2, 4, 1, 0.0),
    "tp8": (8, 1, 1, 0.0),
    "tp2-pp2-cp2-offload50": (2, 2, 2, 0.5),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--data-path", type=Path, help="Optional JSONL split directory for full-context stress inputs")
    parser.add_argument("--launch-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.launch_config:
        cfg = yaml.safe_load(args.launch_config.read_text())
        command = default_training_command(cfg)
        command[-1] = str(ROOT / "tests/run_distill_capacity_worker.py")
        return launch(args.launch_config, command=command,
                      pool_command=[sys.executable, str(ROOT / "tests/run_parallel_distill_frozen_pool.py"),
                                    "--rwkv-config", str(args.launch_config)])
    if args.steps < 3:
        parser.error("--steps must be at least three to measure initialized Adam")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / "configs/config-qwen-2p7b-distill-tp2-sp-pp2-cp2-smoke.yaml").read_text())
    model = yaml.safe_load((ROOT / "configs/config-rwkv7-qwen-distill-single.yaml").read_text())
    base["language_model"] = model["language_model"]
    base.update(load_pretrained_rwkv=model["load_pretrained_rwkv"], global_batch_size=2,
                eval_global_batch_size=2, micro_batch_size=1, eval_micro_batch_size=1,
                train_iters=args.steps, lr_decay_iters=args.steps, eval_interval=args.steps,
                eval_iters=1, save=None, load=None, use_distributed_optimizer=True,
                empty_unused_memory_level=2, rwkv_teacher_save_on_failure=False,
                rwkv_teacher_model="local-synthetic-top100",
                rwkv_teacher_prefetch_lookahead=8, rwkv_teacher_pool_queue_size=16,
                rwkv_teacher_pool_max_results=16, wandb_project="rwkv-qwen-distill-28b")
    base["distributed_timeout_minutes"] = 30  # Cold 28B Triton compilation can exceed ten minutes.
    if args.data_path:
        base.update(data_path=[str(args.data_path.resolve())], rwkv_jsonl_chunk_chars=65536)
    report = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    for name in args.cases:
        folder = out / name
        folder.mkdir(exist_ok=False)
        cfg = copy.deepcopy(base)
        tp, pp, cp, offload = CASES[name]
        if name in ("tp2-pp4-balanced", "tp2-pp4-balanced18"):
            # Embedding/head occupy endpoint stages. Keep fewer RWKV layers
            # there: [20, 24, 24, 20], rather than [22, 22, 22, 22].
            endpoint_layers = 18 if name.endswith("18") else 20
            cfg["language_model"].update(num_layers_in_first_pipeline_stage=endpoint_layers,
                                          num_layers_in_last_pipeline_stage=endpoint_layers)
        cfg["model_parallel"].update(tensor_model_parallel_size=tp,
                                     pipeline_model_parallel_size=pp, context_parallel_size=cp)
        cfg.update(optimizer_cpu_offload=bool(offload), optimizer_offload_fraction=offload,
                   # The checkpoint has 248320 rows. Megatron multiplies this
                   # padding factor by TP; 256 would expand TP4/TP8 vocabularies.
                   make_vocab_size_divisible_by=128 if tp <= 4 else 64,
                   use_precision_aware_optimizer=bool(offload), use_torch_optimizer_for_cpu_offload=bool(offload),
                   overlap_cpu_optimizer_d2h_h2d=False, rwkv_teacher_pool_port=56361,
                   wandb_exp_name=f"28b-seq4096-{name}-local-capacity-{out.name}",
                   tensorboard_dir=str(folder / "tensorboard"), wandb_save_dir=str(folder / "wandb"),
                   rwkv_teacher_sample_records=str(folder / "sample-selection"))
        path = folder / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        runid = uuid.uuid4().hex[:8]
        env = dict(os.environ, WANDB_MODE="online", WANDB_RUN_ID=runid,
                   WANDB_TAGS="distill,28b,capacity,synthetic-teacher",
                   PYTORCH_ALLOC_CONF="expandable_segments:True", TORCH_NCCL_AVOID_RECORD_STREAMS="1",
                   RWKV_CAPACITY_OUTPUT=str(folder), NCCL_NVLS_ENABLE="0")
        peaks, samples = [0] * 8, []
        started = time.monotonic()
        with (folder / "training.log").open("w") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                        "--output", str(out), "--launch-config", str(path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                data = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                                "--format=csv,noheader,nounits"], text=True, timeout=10)
                snapshot = [list(map(int, line.split(","))) for line in data.strip().splitlines()]
                for index, memory, _ in snapshot:
                    peaks[index] = max(peaks[index], memory)
                samples.append(dict(seconds=time.monotonic() - started, gpus=snapshot))
                if time.monotonic() - started > 2400:
                    process.terminate()
                    process.wait(timeout=45)
                    raise TimeoutError(f"28B probe exceeded 40 minutes: {folder}")
                time.sleep(0.5)
        (folder / "gpu-samples.json").write_text(json.dumps(samples) + "\n")
        item = dict(passed=False, exit_code=process.returncode, wall_seconds=time.monotonic() - started,
                    layout=dict(tp=tp, pp=pp, cp=cp, dp=1, sp=True), offload_fraction=offload,
                    peak_gpu_memory_mib=peaks, sequence_length=4096, global_batch_size=2,
                    teacher="synthetic top100 with full-context assertion", wandb_id=runid,
                    log=str(folder / "training.log"))
        report[name] = item
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        if process.returncode:
            item["oom"] = "out of memory" in (folder / "training.log").read_text(errors="replace").lower()
            (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print("FAILED CAPACITY", name, json.dumps(item), flush=True)
            if not item["oom"]:
                raise RuntimeError(f"Non-OOM failure: {folder / 'training.log'}")
            continue
        item["audit"] = audit_training_log(folder / "training.log", 0)
        assert item["audit"]["last_iteration"] == args.steps
        rows = [[json.loads(line) for line in (folder / f"rank{rank}.jsonl").read_text().splitlines()]
                for rank in range(8)]
        for rank, records in enumerate(rows):
            updates = [r for r in records if r["event"] == "optimizer_update"]
            steps = [r for r in records if r["event"] == "train_step"]
            assert len(updates) == len(steps) == args.steps, (name, rank)
            assert all(r["skipped"] == 0 and math.isfinite(r["grad_norm"]) for r in steps)
            assert any(sample["max_abs_delta"] > 0 for row in updates[:2]
                       for sample in row["sampled_master_updates"]), (name, rank, "No sampled master update")
        item["mean_step_seconds_after_first_two"] = statistics.mean(
            r["seconds"] for records in rows for r in records if r["event"] == "train_step" and r["iteration"] > 2)
        item["max_cuda_allocated_mib"] = max(r["peak_allocated_mib"] for records in rows for r in records)
        remote = wandb.Api().run(f"mobius/rwkv-qwen-distill-28b/{runid}")
        assert remote.state == "finished", remote.state
        item.update(passed=True, all_ranks_updated=True, wandb=remote.url,
                    validation_ce=remote.summary.get("distill/validation_student_ce_loss"),
                    validation_ppl=remote.summary.get("distill/validation_student_ppl"))
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PASS CAPACITY", name, json.dumps(item), flush=True)
    return 0 if any(row["passed"] for row in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
