"""Three real 2.7B CP+TP+PP+SP steps with the shared HTTP pool and async targets."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import wandb
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-remaining-20261026"
folder = OUT / "cp-online"
folder.mkdir(exist_ok=True)
cfg = yaml.safe_load(
    (ROOT / "configs/config-qwen-2p7b-distill-tp2-sp-pp2-smoke.yaml").read_text()
)
cfg["model_parallel"].update(data_parallel_size=1, context_parallel_size=2)
cfg.update(
    deterministic_mode=True,
    seq_length=512,
    rwkv_jsonl_chunk_chars=4096,
    global_batch_size=8,
    eval_global_batch_size=8,
    eval_iters=1,
    train_iters=3,
    save=None,
    load=None,
    wandb_entity="mobius",
    wandb_project="rwkv-qwen-distill-2p7b",
    wandb_exp_name="2p7b-cp2-tp2-pp2-sp-online-20261026",
    wandb_save_dir=str(folder / "wandb"),
    tensorboard_dir=str(folder / "tensorboard"),
)
path = OUT / "cp-online.yaml"
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
runid = uuid.uuid4().hex[:8]
with (OUT / "cp-online.log").open("w") as log:
    result = subprocess.run(
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
            WANDB_TAGS="distill,2p7b,cp,tp,pp,sp,online",
        ),
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
assert result.returncode == 0, result.returncode
text = (OUT / "cp-online.log").read_text()
lines = [line for line in text.splitlines() if "number of skipped iterations:" in line]
assert len(lines) == 3, lines
assert all(
    "number of skipped iterations:   0" in line
    and "number of nan iterations:   0" in line
    for line in lines
)
remote = wandb.Api().run("mobius/rwkv-qwen-distill-2p7b/" + runid)
assert remote.state == "finished", remote.state
assert (
    "teacher/consumed" in remote.summary
), "CP0 prefetch telemetry must reach last global rank"
assert remote.summary.get("teacher/failed", 0) == 0, dict(remote.summary)
report = dict(
    layout="TP2 PP2 CP2 SP",
    steps=3,
    seq_length=512,
    global_batch_size=8,
    real_online_teacher=True,
    async_prefetch=True,
    shared_pool=True,
    no_nan_or_skip=True,
    wandb=remote.url,
    verified_finished=True,
    validation_ce=remote.summary.get("distill/validation_student_ce_loss"),
    validation_ppl=remote.summary.get("distill/validation_student_ppl"),
    prefetch_telemetry_visible=True,
)
(OUT / "context-online.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)
