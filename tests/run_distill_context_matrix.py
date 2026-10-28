"""Contiguous CP distillation through full-context teacher, DDP/TP/PP and restart."""

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
REFERENCE = ROOT / "training-smoke/dp2-distill-20261023/dense-bf16"
SEQUENCE_LENGTH = 128
TEACHER_SAFETY = False
CASES = {
    "dense": (1, 1, 1, False, 1, False),
    "cp2": (1, 1, 1, False, 2, False),
    "dp2-cp2": (2, 1, 1, False, 2, False),
    "tp2-cp2": (1, 2, 1, False, 2, False),
    "tp2-sp-cp2": (1, 2, 1, True, 2, False),
    "pp2-cp2": (1, 1, 2, False, 2, False),
    "tp2-sp-pp2-cp2": (1, 2, 2, True, 2, False),
    "tp2-sp-pp2-cp2-recompute": (1, 2, 2, True, 2, True),
    "cp4": (1, 1, 1, False, 4, False),
    "dp2-cp4": (2, 1, 1, False, 4, False),
    "tp2-cp4": (1, 2, 1, False, 4, False),
    "tp2-sp-cp4": (1, 2, 1, True, 4, False),
    "pp2-cp4": (1, 1, 2, False, 4, False),
    "tp2-sp-cp4-recompute": (1, 2, 1, True, 4, True),
    "cp8": (1, 1, 1, False, 8, False),
}


def run(name, resume):
    dp, tp, pp, sp, cp, recompute = CASES[name]
    cfg = yaml.safe_load(
        (ROOT / "training-smoke/dp2-distill-20261023/dense-bf16.yaml").read_text()
    )
    cfg["model_parallel"].update(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        sequence_parallel=sp,
    )
    if recompute:
        cfg["language_model"].update(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
    folder = OUT / (name + ("-resume" if resume else ""))
    folder.mkdir(exist_ok=True)
    cfg.update(
        deterministic_mode=True,
        save=str(folder / "checkpoints"),
        load=None,
        train_iters=3,
        seq_length=SEQUENCE_LENGTH,
        distributed_timeout_minutes=10,
        eval_iters=1,
        eval_interval=3,
        wandb_project=None,
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
    with (OUT / (folder.name + ".log")).open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={dp*tp*pp*cp}",
                str(ROOT / "tests/run_dp_distill_training.py"),
                "--rwkv-config",
                str(path),
            ],
            cwd=ROOT,
            env=dict(
                os.environ,
                RWKV_DP_AUDIT_DIR=str(folder),
                RWKV_DP_OPTIMIZER_AUDIT="1",
                RWKV_DP_FP32="0",
                RWKV_TEACHER_SAFETY=str(int(TEACHER_SAFETY)),
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    assert result.returncode == 0, (folder.name, result.returncode)
    print("COMPLETE CONTEXT", folder.name, flush=True)


def check(name):
    dp, tp, pp, sp, cp, recompute = CASES[name]

    def records(folder, step):
        return [
            torch.load(OUT / folder / f"step{step}-rank{rank}.pt", weights_only=False)
            for rank in range(dp * tp * pp * cp)
        ]

    def reference(step):
        return [
            torch.load(
                REFERENCE / f"step{step}-rank0.pt",
                weights_only=False,
            )
        ]

    for step in (1, 2, 3):
        rows = records(name, step)
        representatives = [r for r in rows if r["cp"] == 0]
        replicas(representatives, dp, tp, pp)
        assert samples(representatives, dp, pp)[0] == samples(reference(step), 1, 1)[0]
        for row in rows:
            assert row["skipped"] == 0
            assert row["teacher_requests"] == (
                16 // dp // 2
                if row["cp"] == 0 and row["tp"] == 0 and row["pp"] == pp - 1
                else 0
            )
            other = next(
                r
                for r in representatives
                if (r["dp"], r["tp"], r["pp"]) == (row["dp"], row["tp"], row["pp"])
            )
            for field in ("before", "grads", "after", "logits", "samples"):
                exact(
                    row[field],
                    other[field],
                    f'CP{row["cp"]}/{field}',
                    dict(tensors=0, elements=0),
                )
    rows = [r for r in records(name, 1) if r["cp"] == 0]
    initial = canonical(rows, "before", tp, pp)
    expected_initial = canonical(reference(1), "before", 1, 1)
    assert initial.keys() == expected_initial.keys()
    assert all(torch.equal(value, expected_initial[key]) for key, value in initial.items()), (
        name, "Initial model weights differ from the CP1 reference")
    grad = canonical(rows, "grads", tp, pp)
    expected = canonical(reference(1), "grads", 1, 1)
    metrics = {k: metric(v, expected[k]) for k, v in grad.items()}
    assert all(m["finite"] for m in metrics.values())
    significant = {k: v for k, v in metrics.items() if v["reference_l2"] >= 1e-3}
    worst = max(significant, key=lambda k: significant[k]["relative_l2"])
    assert significant[worst]["relative_l2"] < 0.05, (name, worst, significant[worst])
    _, actual_logits = samples(rows, dp, pp)
    keys, expected_logits = samples(reference(1), 1, 1)
    keys = sorted(keys)
    logits = metric(
        torch.stack([actual_logits[k] for k in keys]),
        torch.stack([expected_logits[k] for k in keys]),
    )
    assert logits["relative_l2"] < 0.02, (name, logits)
    final = next(r for r in rows if r["tp"] == 0 and r["dp"] == 0 and r["pp"] == pp - 1)
    losses = {
        key: abs(final["losses"][key] - reference(1)[0]["losses"][key])
        / abs(reference(1)[0]["losses"][key])
        for key in ("lm loss", "distill/kl_loss", "distill/student_ce_loss")
    }
    assert max(losses.values()) < 0.005, (name, losses)
    counts = dict(tensors=0, elements=0)
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
            exact(a[field], b[field], f"rank{rank}/{field}", counts)
    return dict(
        layout=dict(dp=dp, tp=tp, pp=pp, sp=sp, cp=cp, recompute=recompute),
        logits=logits,
        loss_relative_errors=losses,
        worst_gradient_parameter=worst,
        worst_gradient=significant[worst],
        strict_resume_bit_identical=True,
        cp_dp_replicas_bit_identical=True,
        full_context_teacher_once=True,
        samples_disjoint_and_complete=True,
        sequence_length=SEQUENCE_LENGTH,
        teacher_prepared_before_forward=TEACHER_SAFETY,
        initial_weights_bit_identical=True,
        exact_resume_counts=counts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=list(CASES),
                        default=[name for name in CASES if name != "dense" and "cp4" not in name and name != "cp8"])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--teacher-safety", action="store_true")
    parser.add_argument("--generate-reference", action="store_true",
                        help="Run a fresh CP1 baseline in output-dir/dense before the matrix")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    SEQUENCE_LENGTH = args.seq_length
    TEACHER_SAFETY = args.teacher_safety
    for name in args.cases:
        cp = CASES[name][4]
        if SEQUENCE_LENGTH < 1 or (cp > 1 and SEQUENCE_LENGTH % (64 * cp)):
            parser.error(f"{name} requires --seq-length divisible by {64 * cp}")
    if args.generate_reference:
        REFERENCE = OUT / "dense"
        if not args.check_only:
            run("dense", False)
    path = OUT / "context.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for name in args.cases:
        try:
            if not args.check_only:
                run(name, False)
                run(name, True)
            report[name] = check(name)
        except Exception as error:
            report[name] = dict(passed=False, error=str(error))
            path.write_text(json.dumps(report, indent=2) + "\n")
            raise
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS CONTEXT", name, report[name], flush=True)
