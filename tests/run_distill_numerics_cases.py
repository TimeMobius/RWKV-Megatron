"""Run selected trace/control experiments in isolated snapshot/checkpoint paths."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-numerics-20261023"
CKPTS = ROOT / "training-smoke/checkpoints/distill-numerics-20261023"


def run(name, dp, tp, pp, sp=False, row=False, oracle=False, resume=False, steps=1):
    base = ROOT / "training-smoke/dp2-distill-20261023/dense-bf16.yaml"
    cfg = yaml.safe_load(base.read_text())
    cfg["model_parallel"].update(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        sequence_parallel=sp,
    )
    cfg.update(
        train_iters=steps,
        eval_iters=1 if "plain" in name else 0,
        save=str(CKPTS / name) if steps == 3 else None,
        wandb_project=None,
        load=None,
    )
    if resume:
        original = name.removesuffix("-resume")
        source = CKPTS / (name + "-source")
        source.mkdir(parents=True, exist_ok=True)
        link = source / "iter_0000002"
        if not link.exists():
            link.symlink_to(CKPTS / original / "iter_0000002", target_is_directory=True)
        (source / "latest_checkpointed_iteration.txt").write_text("2")
        cfg["load"] = str(source)
    if "original" in name and "rewarm" not in name:
        tag = "dp2-tp2-sp-pp2-bf16" if sp else "dp2-tp2-pp2-bf16"
        cfg["load"] = str(
            ROOT
            / "training-smoke/checkpoints/dp2-distill-20261023"
            / (tag + "-resume-source")
        )
        cfg["eval_iters"] = 1
    config = OUT / f"{name}.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    folder = OUT / name
    folder.mkdir(exist_ok=True, parents=True)
    level = next(
        (
            v
            for v in ("none", "sync", "single-thread", "eager-mix", "plain", "cold")
            if v in name
        ),
        "full",
    )
    env = dict(
        os.environ,
        RWKV_DP_AUDIT_DIR=str(folder),
        DIAG_ROW_FP32=str(int(row)),
        RWKV_DP_FP32=str(int(oracle)),
        DIAG_TRACE_LEVEL=level,
    )
    if "rewarm" in name:
        tag = "dp2-tp2-sp-pp2-bf16" if sp else "dp2-tp2-pp2-bf16"
        env["DIAG_REWARM_CHECKPOINT"] = str(
            ROOT
            / "training-smoke/checkpoints/dp2-distill-20261023"
            / (tag + "-resume-source")
        )
        cfg["eval_iters"] = 1
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if "cold" in name:
        env.update(
            TRITON_CACHE_DIR=str(folder / "triton"),
            TORCHINDUCTOR_CACHE_DIR=str(folder / "inductor"),
        )
    with (OUT / f"{name}.log").open("w") as log:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={dp*tp*pp}",
                str(
                    ROOT
                    / (
                        "tests/run_dp_distill_training.py"
                        if level == "plain"
                        else "tests/trace_distill_numerics.py"
                    )
                ),
                "--rwkv-config",
                str(config),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert proc.returncode == 0, (name, proc.returncode)
    print("TRACE COMPLETE", name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {
        "dense": (1, 1, 1),
        "tp2": (1, 2, 1),
        "dp2-tp2": (2, 2, 1),
        "tp2-sp-none": (1, 2, 1, True),
        "tp2-sp-row32-none": (1, 2, 1, True, True),
        "dense-row32": (1, 1, 1, False, True),
        "tp2-row32": (1, 2, 1, False, True),
        "dense-oracle": (1, 1, 1, False, False, True),
        "tp2-oracle": (1, 2, 1, False, False, True),
        "combined": (2, 2, 2),
        "combined-resume": (2, 2, 2),
        "combined-sp-plain": (2, 2, 2, True),
        "combined-sp-plain-resume": (2, 2, 2, True),
        "combined-plain": (2, 2, 2),
        "combined-plain-resume": (2, 2, 2),
        "combined-eager-mix": (2, 2, 2),
        "combined-eager-mix-resume": (2, 2, 2),
        "combined-cold": (2, 2, 2),
        "combined-cold-resume": (2, 2, 2),
        "combined-sp-cold": (2, 2, 2, True),
        "combined-sp-cold-resume": (2, 2, 2, True),
        "combined-original-none": (2, 2, 2),
        "combined-sp-original-none": (2, 2, 2, True),
        "combined-original": (2, 2, 2),
        "combined-sp-original": (2, 2, 2, True),
        "combined-original-rewarm": (2, 2, 2),
        "combined-sp-original-rewarm": (2, 2, 2, True),
        "combined-none": (2, 2, 2),
        "combined-none-resume": (2, 2, 2),
        "combined-sync": (2, 2, 2),
        "combined-sync-resume": (2, 2, 2),
        "combined-single-thread": (2, 2, 2),
        "combined-single-thread-resume": (2, 2, 2),
    }
    for name in args.cases or list(cases):
        run(
            name,
            *cases[name],
            resume=name.endswith("-resume"),
            steps=3 if name.startswith("combined") else 1,
        )
