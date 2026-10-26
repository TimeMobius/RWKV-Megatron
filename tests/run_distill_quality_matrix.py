"""Paired real-Qwen target replay for 2.7B checkpoint quality and GPU throughput."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-remaining-20261026"
LAYOUTS = dict(
    dense=(1, 1, 1, False),
    tp2=(1, 2, 1, False),
    tp2sp=(1, 2, 1, True),
    pp2=(1, 1, 2, False),
    combined=(2, 2, 2, True),
)


def run(name, layout, record=False, load=None, initial=False, cp=1):
    dp, tp, pp, sp = layout
    cfg = yaml.safe_load(
        (ROOT / "configs/config-qwen-2p7b-distill-tp2-sp-pp2-smoke.yaml").read_text()
    )
    cfg["model_parallel"].update(
        data_parallel_size=dp,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        sequence_parallel=sp,
        context_parallel_size=cp,
    )
    cfg.update(
        deterministic_mode=True,
        seq_length=512,
        rwkv_jsonl_chunk_chars=4096,
        global_batch_size=8,
        eval_global_batch_size=8,
        train_iters=8,
        lr_decay_iters=8,
        eval_iters=4,
        eval_interval=8,
        save_interval=8,
        load=load,
        save=None if (record or initial or load) else str(OUT / name / "checkpoints"),
        rwkv_teacher_pool=False,
        rwkv_teacher_async_prefetch=False,
        wandb_entity="mobius",
        wandb_project="rwkv-qwen-distill-2p7b",
        wandb_exp_name=f"paired-2p7b-{name}-20261026",
        wandb_save_dir=str(OUT / name / "wandb"),
        tensorboard_dir=str(OUT / name / "tensorboard"),
        log_interval=1,
        num_workers=0,
    )
    if initial:
        cfg.update(train_iters=0, skip_train=True)
    if load:
        cfg.update(skip_train=True, no_load_optim=True, no_load_rng=True)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    path = OUT / (name + ".yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    runid = uuid.uuid4().hex[:8]
    env = dict(
        os.environ,
        RWKV_EXT_OUTPUT=str(folder),
        RWKV_EXT_TEACHER_CACHE=str(OUT / "teacher-cache"),
        RWKV_EXT_RECORD=str(int(record)),
        WANDB_RUN_ID=runid,
        WANDB_MODE="online",
        WANDB_TAGS="distill,2p7b,paired-targets,real-qwen-recorded,quality-validation",
    )
    with (OUT / (name + ".log")).open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={dp*tp*pp*cp}",
                str(ROOT / "tests/run_distill_cached_training.py"),
                "--rwkv-config",
                str(path),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert result.returncode == 0, (name, result.returncode)
    import wandb

    remote = wandb.Api().run("mobius/rwkv-qwen-distill-2p7b/" + runid)
    assert remote.state == "finished", remote.state
    info = dict(
        layout=dict(dp=dp, tp=tp, pp=pp, sp=sp, cp=cp),
        wandb=remote.url,
        verified_finished=True,
        validation_ce=remote.summary.get("distill/validation_student_ce_loss"),
        validation_ppl=remote.summary.get("distill/validation_student_ppl"),
        save=cfg["save"],
        load=load,
    )
    path = OUT / "quality-runs.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = info
    path.write_text(json.dumps(data, indent=2) + "\n")
    print("COMPLETE QUALITY", name, info, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "record",
            "initial",
            "dense",
            "tp2",
            "tp2sp",
            "pp2",
            "combined",
            "dense-reload",
            "combined-reload",
        ],
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        if name == "record":
            run(name, LAYOUTS["dense"], record=True)
        elif name == "initial":
            run(name, LAYOUTS["dense"], record=True, initial=True)
        elif name == "cp2-checkpoint":
            run(name, LAYOUTS["dense"], load=str(OUT / "dense/checkpoints"), cp=2)
        elif name == "cp-combined-checkpoint":
            run(name, (1, 2, 2, True), load=str(OUT / "combined/checkpoints"), cp=2)
        elif name.endswith("-reload"):
            source = name.removesuffix("-reload")
            run(name, LAYOUTS[source], load=str(OUT / source / "checkpoints"))
        else:
            run(name, LAYOUTS[name])
