"""Verify real Qwen teacher training in all certified layouts and upload evidence.

Run through training-smoke/parallel-distill-20261023/run.sh for CUDA settings.
Each layout runs the unchanged production launcher, pool, prefetch and trainer.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import wandb
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/parallel-distill-20261023"
LAYOUTS = ("dense", "tp2", "tp2-sp", "pp2", "tp2-sp-pp2", "tp4-sp-pp2")


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = OUT / f"online-verification-{stamp}"
    folder.mkdir(parents=True)
    result = dict(
        started_at=datetime.now().isoformat(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        teacher="real external Qwen3.8-27B; production HTTP, pool and async prefetch",
        folder=str(folder),
        layouts={},
    )
    summary = OUT / "online-verification.json"
    for layout in LAYOUTS:
        cfg_path = ROOT / f"configs/config-qwen-2p7b-distill-{layout}-smoke.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        run_dir = folder / layout
        run_dir.mkdir()
        cfg.update(
            wandb_exp_name=f"qwen-2p7b-real-online-{layout}-{stamp}",
            wandb_save_dir=str(run_dir / "wandb"),
            tensorboard_dir=str(run_dir / "tensorboard"),
        )
        config = run_dir / "config.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        run_id = uuid.uuid4().hex[:8]
        env = dict(
            os.environ,
            WANDB_MODE="online",
            WANDB_RUN_ID=run_id,
            WANDB_TAGS="distill,parallel-validation,real-qwen,online-teacher",
        )
        start = time.monotonic()
        with (run_dir / "train.log").open("w") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "tools/launch_distillation.py",
                    "--rwkv-config",
                    str(config),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        text = (run_dir / "train.log").read_text()
        info = dict(
            exitcode=process.returncode,
            seconds=time.monotonic() - start,
            wandb=f"http://dian.sea-group.org:9098/mobius/rwkv-qwen-distill-2p7b/runs/{run_id}",
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            steps=[],
        )
        result["layouts"][layout] = info
        summary.write_text(json.dumps(result, indent=2) + "\n")
        assert process.returncode == 0, (
            layout,
            process.returncode,
            str(run_dir / "train.log"),
        )
        lines = [
            line
            for line in text.splitlines()
            if re.search(r"iteration\s+[123]/\s+3\s+\|", line)
        ]
        assert len(lines) == 3, (layout, len(lines))
        for line in lines:

            def number(key):
                return float(re.search(re.escape(key) + r":\s+([\d.Ee+-]+)", line)[1])

            step = {
                key: number(key)
                for key in (
                    "lm loss",
                    "distill/kl_loss",
                    "distill/student_ce_loss",
                    "grad norm",
                    "elapsed time per iteration (ms)",
                    "memory/max_allocated_mib",
                    "number of skipped iterations",
                    "number of nan iterations",
                    "teacher/failed",
                    "teacher/completed",
                    "teacher/consumed",
                )
            }
            assert (
                step["number of skipped iterations"]
                == step["number of nan iterations"]
                == 0
            )
            assert step["teacher/failed"] == 0
            info["steps"].append(step)
        assert "'submitted': 12, 'consumed': 12, 'completed': 12, 'failed': 0" in text
        assert "on validation set" in text and "on test set" in text
        info["train_teacher_microbatches"] = 12
        info["validation_and_test_completed"] = True
        remote = wandb.Api().run(f"mobius/rwkv-qwen-distill-2p7b/{run_id}")
        assert remote.state == "finished", (layout, remote.state)
        info["wandb_verified_finished"] = True
        for key in (
            "distill/validation_student_ce_loss",
            "distill/test_student_ce_loss",
            "teacher/failed",
        ):
            assert key in remote.summary, (layout, key)
            info[key] = remote.summary[key]
        summary.write_text(json.dumps(result, indent=2) + "\n")
        print(f"PASS REAL ONLINE {layout}: {info['wandb']}", flush=True)
    result["finished_at"] = datetime.now().isoformat()
    result["all_passed"] = True
    summary.write_text(json.dumps(result, indent=2) + "\n")
    upload_report(result)


def upload_report(result):
    assert result["all_passed"] and len(result["layouts"]) == 6
    folder = Path(result["folder"])
    stamp = folder.name.removeprefix("online-verification-")
    summary = OUT / "online-verification.json"
    reference = result["layouts"]["dense"]["steps"][0]
    for layout, info in result["layouts"].items():
        errors = {
            key: abs(info["steps"][0][key] - reference[key]) / abs(reference[key])
            for key in ("distill/kl_loss", "distill/student_ce_loss", "grad norm")
        }
        # Independently requested teacher responses use the same seed and inputs.
        # Frozen-teacher tests remain the controlled gradient alignment evidence.
        for key, error in errors.items():
            assert error < (0.05 if key == "grad norm" else 0.005), (layout, key, error)
        info["first_step_relative_errors"] = errors
    summary.write_text(json.dumps(result, indent=2) + "\n")
    run = wandb.init(
        entity="mobius",
        project="rwkv-qwen-distill-2p7b",
        name=f"real-qwen-parallel-certification-{stamp}",
        dir=str(folder),
        tags=["parallel-validation", "real-qwen", "online-teacher"],
        config=dict(
            git_commit=result["git_commit"],
            steps=3,
            global_batch_size=4,
            teacher=result["teacher"],
            dp=1,
            cp=1,
        ),
    )
    table = wandb.Table(
        columns=[
            "layout",
            "step",
            "kl",
            "student_ce",
            "grad_norm",
            "time_ms",
            "peak_mib",
            "run",
        ]
    )
    for layout, info in result["layouts"].items():
        for idx, step in enumerate(info["steps"], 1):
            table.add_data(
                layout,
                idx,
                step["distill/kl_loss"],
                step["distill/student_ce_loss"],
                step["grad norm"],
                step["elapsed time per iteration (ms)"],
                step["memory/max_allocated_mib"],
                info["wandb"],
            )
    run.log({"real_online_training": table})
    run.summary.update(
        dict(
            layouts_passed=6,
            nan_iterations=0,
            skipped_iterations=0,
            teacher_failed_microbatches=0,
            online_teacher_success_certified=True,
        )
    )
    artifact = wandb.Artifact(
        f"real-qwen-parallel-validation-{stamp}", type="validation-report"
    )
    artifact.add_file(str(summary))
    for path in folder.glob("*/config.yaml"):
        artifact.add_file(str(path), name=f"{path.parent.name}/config.yaml")
    run.log_artifact(artifact).wait()
    url = run.url
    run.finish()
    assert (
        wandb.Api().run("mobius/rwkv-qwen-distill-2p7b/" + url.rsplit("/", 1)[1]).state
        == "finished"
    )
    result["report_wandb"] = url
    result["report_verified_finished"] = True
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS ALL REAL ONLINE LAYOUTS: {url}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--report-only"]:
        upload_report(json.loads((OUT / "online-verification.json").read_text()))
    else:
        main()
