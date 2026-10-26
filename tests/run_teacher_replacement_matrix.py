"""Production DP/TP/PP/SP/CP schedules keep valid replacement samples aligned."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from compare_dp_distill import replicas, samples
from compare_dp_distill_resume import exact
from compare_teacher_replacement import precision

ROOT = Path(__file__).resolve().parents[1]
POOL_MODE = os.environ.get("RWKV_TEST_POOL_REPLACEMENT") == "1"
OUT = ROOT / ("training-smoke/distill-remaining-20261026/teacher-pool-replacement"
              if POOL_MODE else "training-smoke/distill-remaining-20261026/teacher-replacement")
OUT.mkdir(exist_ok=True)
CASES = {
    "dense": (1, 1, 1, False, 1),
    "combined": (2, 2, 2, True, 1),
    "cp-combined": (1, 2, 2, True, 2),
}
report = {}
for name, (dp, tp, pp, sp, cp) in CASES.items():
    for resume in (False, True):
        folder = OUT / (name + ("-resume" if resume else ""))
        folder.mkdir(exist_ok=True)
        cfg = yaml.safe_load(
            (ROOT / "training-smoke/dp2-distill-20261023/dense-bf16.yaml").read_text()
        )
        cfg["model_parallel"].update(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            sequence_parallel=sp,
            context_parallel_size=cp,
        )
        cfg.update(
            deterministic_mode=True,
            save=str(folder / "checkpoints"),
            load=None,
            wandb_project=None,
            train_iters=3,
            eval_interval=3,
            eval_iters=1,
        )
        if POOL_MODE:
            cfg.update(rwkv_teacher_pool=True, rwkv_teacher_replace_bad_samples=True,
                       rwkv_teacher_async_prefetch=True)
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
        path = folder / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        pool = None
        selection = OUT / (name + "-selection")
        environment = dict(
            os.environ, RWKV_DP_AUDIT_DIR=str(folder), RWKV_DP_OPTIMIZER_AUDIT="1",
            RWKV_REPLACEMENT_DIR=str(selection),
        )
        if POOL_MODE:
            from teacher_sample_pool_fixture import write_fixture

            selection.mkdir(exist_ok=True)
            for split in range(3):
                source = selection / f"split{split}.jsonl"
                if not source.exists():
                    write_fixture(source, split)
            port_file = folder / "pool-port.txt"
            port_file.unlink(missing_ok=True)
            pool_log = (folder / "pool.log").open("w")
            pool = subprocess.Popen(
                [sys.executable, str(ROOT / "tests/run_teacher_sample_pool_fixture.py"), str(port_file)],
                cwd=ROOT, stdout=pool_log, stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 30
            while not port_file.exists():
                assert pool.poll() is None and time.monotonic() < deadline, "fixture pool startup failed"
                time.sleep(0.05)
            environment["RWKV_REPLACEMENT_POOL_PORT"] = port_file.read_text().strip()
        try:
            with (folder / "training.log").open("w") as log:
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
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                )
        finally:
            if pool is not None:
                pool.terminate()
                pool.wait(timeout=10)
                pool_log.close()
        assert result.returncode == 0, (name, resume, result.returncode)
        print("COMPLETE", folder.name, flush=True)

    def records(tag, step):
        return [
            torch.load(OUT / tag / f"step{step}-rank{rank}.pt", weights_only=False)
            for rank in range(dp * tp * pp * cp)
        ]

    for step in (1, 2, 3):
        rows = records(name, step)
        rows = [r for r in rows if r["cp"] == 0]
        replicas(rows, dp, tp, pp)
        actual, _ = samples(rows, dp, pp)
        if name != "dense":
            dense = [
                torch.load(OUT / "dense" / f"step{step}-rank0.pt", weights_only=False)
            ]
            assert actual == samples(dense, 1, 1)[0]
        assert all(row["skipped"] == 0 for row in rows)
    counts = dict(tensors=0, elements=0)
    for rank, (a, b) in enumerate(zip(records(name, 3), records(name + "-resume", 3))):
        for field in [
            "samples",
            "before",
            "grads",
            "after",
            "optimizer",
            "logits",
            "grad_norm",
        ]:
            exact(a[field], b[field], f"{rank}/{field}", counts)
    logs = list((OUT / (name + "-selection")).glob("rejected-samples.jsonl"))
    rejected = [
        json.loads(line) for path in logs for line in path.read_text().splitlines()
    ]
    assert rejected
    report[name] = dict(
        layout=dict(dp=dp, tp=tp, pp=pp, sp=sp, cp=cp),
        steps=3,
        strict_resume_bit_identical=True,
        global_samples_aligned=True,
        rejected_samples=len(rejected),
        no_nan_or_optimizer_skip=True,
    )
    if POOL_MODE and name != "dense":
        report[name]["precision"] = precision(OUT, name, dp, tp, pp)
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", name, report[name], flush=True)
