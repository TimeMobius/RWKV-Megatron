"""Check DP4 and optimizer variants through real DDP, Adam and checkpoint replay."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from compare_dp_distill import canonical, metric, replicas, samples
from compare_dp_distill_resume import exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-remaining-20261026"
CASES = {
    "distopt-dp2": (2, 1, 1, False, True, 0.0),
    "distopt-combined": (2, 2, 2, True, True, 0.0),
    "offload-dp2": (2, 1, 1, False, False, 0.5),
    "offload-combined": (2, 2, 2, True, False, 0.5),
    "distopt-offload-combined": (2, 2, 2, True, True, 0.5),
    "dp4": (4, 1, 1, False, False, 0.0),
    "dp4-tp2-sp": (4, 2, 1, True, False, 0.0),
    "dp4-pp2": (4, 1, 2, False, False, 0.0),
}


def run(name, resume):
    dp, tp, pp, sp, distributed, offload = CASES[name]
    cfg = yaml.safe_load(
        (ROOT / "training-smoke/dp2-distill-20261023/dense-bf16.yaml").read_text()
    )
    cfg["model_parallel"].update(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        sequence_parallel=sp,
        gradient_accumulation_fusion=False,
    )
    folder = OUT / (name + ("-resume" if resume else ""))
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = folder / "checkpoints"
    cfg.update(
        deterministic_mode=True,
        use_distributed_optimizer=distributed,
        optimizer_cpu_offload=offload > 0,
        optimizer_offload_fraction=offload,
        use_torch_optimizer_for_cpu_offload=offload > 0,
        overlap_cpu_optimizer_d2h_h2d=False,
        save=str(ckpt),
        load=None,
        train_iters=3,
        eval_iters=1,
        eval_interval=3,
        wandb_project=None,
        ckpt_format="torch",
    )
    if resume:
        source = folder / "source"
        source.mkdir(exist_ok=True)
        link = source / "iter_0000002"
        if not link.exists():
            link.symlink_to(
                OUT / name / "checkpoints/iter_0000002", target_is_directory=True
            )
        (source / "latest_checkpointed_iteration.txt").write_text("2")
        cfg["load"] = str(source)
    path = OUT / (folder.name + ".yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(
        os.environ,
        RWKV_DP_AUDIT_DIR=str(folder),
        RWKV_DP_OPTIMIZER_AUDIT="1",
        RWKV_DP_FP32="0",
    )
    with (OUT / (folder.name + ".log")).open("w") as log:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={dp*tp*pp}",
                str(ROOT / "tests/run_dp_distill_training.py"),
                "--rwkv-config",
                str(path),
            ],
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    assert proc.returncode == 0, (folder.name, proc.returncode)
    print("COMPLETE OPTIMIZER", folder.name, flush=True)


def check(name):
    dp, tp, pp, sp, distributed, offload = CASES[name]

    def records(folder, step):
        return [
            torch.load(OUT / folder / f"step{step}-rank{rank}.pt", weights_only=False)
            for rank in range(dp * tp * pp)
        ]

    reference = [
        torch.load(
            ROOT / "training-smoke/dp2-distill-20261023/dense-bf16/step1-rank0.pt",
            weights_only=False,
        )
    ]
    expected = canonical(reference, "grads", 1, 1)
    count = dict(tensors=0, elements=0)
    for step in (1, 2, 3):
        rows = records(name, step)
        replicas(rows, dp, tp, pp)
        wanted = [
            torch.load(
                ROOT
                / f"training-smoke/dp2-distill-20261023/dense-bf16/step{step}-rank0.pt",
                weights_only=False,
            )
        ]
        assert samples(rows, dp, pp)[0] == samples(wanted, 1, 1)[0]
        assert all(row["skipped"] == 0 for row in rows)
    first = records(name, 1)
    grad = canonical(first, "grads", tp, pp)
    metrics = {k: metric(v, expected[k]) for k, v in grad.items()}
    significant = {k: v for k, v in metrics.items() if v["reference_l2"] >= 1e-3}
    worst = max(significant, key=lambda k: significant[k]["relative_l2"])
    assert significant[worst]["relative_l2"] < 0.05, (name, worst, significant[worst])
    for rank, (a, b) in enumerate(zip(records(name, 3), records(name + "-resume", 3))):
        for field in (
            "samples",
            "before",
            "grads",
            "after",
            "logits",
            "optimizer",
            "grad_norm",
            "skipped",
        ):
            exact(a[field], b[field], f"rank{rank}/{field}", count)
    return dict(
        layout=dict(dp=dp, tp=tp, pp=pp, sp=sp),
        distributed_optimizer=distributed,
        cpu_offload_fraction=offload,
        worst_gradient_parameter=worst,
        worst_gradient=significant[worst],
        strict_resume_bit_identical=True,
        dp_replicas_bit_identical=True,
        samples_disjoint_and_complete=True,
        exact_resume_counts=count,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "optimizer-dp4.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in args.cases:
        try:
            run(name, False)
            run(name, True)
            report[name] = check(name)
        except Exception as error:
            report[name] = dict(passed=False, error=str(error))
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            raise
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS OPTIMIZER", name, report[name], flush=True)
