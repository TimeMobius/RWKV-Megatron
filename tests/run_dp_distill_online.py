"""2.7B real-teacher DP smoke with fixed global batch and explicit torchrun size."""

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime

import wandb
import yaml
from prepare_dp_distill import OUT, ROOT


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = OUT / f"online-{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(started_at=datetime.now().isoformat(), folder=str(folder), layouts={})
    for name, source, dp in [
        ("dense", "dense", 1),
        ("dp2", "dense", 2),
        ("dp2-tp2-sp-pp2", "tp2-sp-pp2", 2),
    ]:
        cfg = yaml.safe_load(
            (ROOT / f"configs/config-qwen-2p7b-distill-{source}-smoke.yaml").read_text()
        )
        cfg.update(
            global_batch_size=8,
            eval_global_batch_size=8,
            rwkv_teacher_pool_port=56225,
            wandb_exp_name=f"qwen-2p7b-real-{name}-{stamp}",
            wandb_save_dir=str(folder / name / "wandb"),
            tensorboard_dir=str(folder / name / "tensorboard"),
        )
        path = folder / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        tp = cfg["model_parallel"]["tensor_model_parallel_size"]
        pp = cfg["model_parallel"]["pipeline_model_parallel_size"]
        runid = uuid.uuid4().hex[:8]
        env = dict(
            os.environ,
            WANDB_MODE="online",
            WANDB_RUN_ID=runid,
            WANDB_TAGS="distill,dp2,real-qwen,parallel-validation",
        )
        command = [
            sys.executable,
            "-u",
            "tools/launch_distillation.py",
            "--rwkv-config",
            str(path),
            "--",
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={dp*tp*pp}",
            str(ROOT / "pretrain_rwkv7.py"),
        ]
        with (folder / f"{name}.log").open("w") as log:
            proc = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        text = (folder / f"{name}.log").read_text()
        assert proc.returncode == 0, (
            name,
            proc.returncode,
            str(folder / f"{name}.log"),
        )
        lines = [
            line
            for line in text.splitlines()
            if re.search(r"iteration\s+[123]/\s+3\s+\|", line)
        ]
        assert len(lines) == 3, (name, len(lines))
        steps = []
        for line in lines:
            step = {
                key: float(re.search(re.escape(key) + r":\s+([\d.Ee+-]+)", line)[1])
                for key in (
                    "distill/kl_loss",
                    "distill/student_ce_loss",
                    "grad norm",
                    "number of skipped iterations",
                    "number of nan iterations",
                    "teacher/failed",
                )
            }
            assert (
                step["number of skipped iterations"]
                == step["number of nan iterations"]
                == step["teacher/failed"]
                == 0
            )
            steps.append(step)
        count = 3 * 8 // dp
        assert (
            text.count(
                f"'submitted': {count}, 'consumed': {count}, 'completed': {count}, 'failed': 0"
            )
            == dp
        ), (name, "teacher stats")
        assert "on validation set" in text and "on test set" in text
        remote = wandb.Api().run(f"mobius/rwkv-qwen-distill-2p7b/{runid}")
        assert remote.state == "finished", remote.state
        result["layouts"][name] = dict(
            dp=dp,
            tp=tp,
            pp=pp,
            exitcode=0,
            steps=steps,
            teacher_microbatches_per_replica=count,
            wandb=remote.url,
            wandb_verified_finished=True,
            validation_ce=remote.summary["distill/validation_student_ce_loss"],
            test_ce=remote.summary["distill/test_student_ce_loss"],
        )
        (OUT / "online.json").write_text(json.dumps(result, indent=2) + "\n")
        print("PASS DP REAL ONLINE", name, remote.url, flush=True)
    reference = result["layouts"]["dense"]["steps"][0]
    for name, info in result["layouts"].items():
        errors = {
            key: abs(info["steps"][0][key] / reference[key] - 1)
            for key in reference
            if reference[key] != 0
        }
        for key, error in errors.items():
            assert error < (0.05 if key == "grad norm" else 0.005), (name, key, error)
        info["first_step_relative_errors"] = errors
    result["all_passed"] = True
    result["finished_at"] = datetime.now().isoformat()
    (OUT / "online.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
