"""Teacher failure must reach CP/TP peers and stop all PP ranks promptly."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-remaining-20261026"
folder = OUT / "cp-teacher-failure"
folder.mkdir(exist_ok=True)
cfg = yaml.safe_load((OUT / "tp2-sp-pp2-cp2.yaml").read_text())
cfg.update(save=None, load=None, train_iters=1)
path = OUT / "cp-teacher-failure.yaml"
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
started = time.monotonic()
with (OUT / "cp-teacher-failure.log").open("w") as log:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(ROOT / "tests/run_dp_distill_training.py"),
            "--rwkv-config",
            str(path),
        ],
        cwd=ROOT,
        env=dict(os.environ, RWKV_CP_FAIL="1", RWKV_DP_AUDIT_DIR=str(folder)),
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
seconds = time.monotonic() - started
text = (OUT / "cp-teacher-failure.log").read_text()
assert result.returncode != 0, result.returncode
assert "injected CP source teacher failure" in text
assert "teacher request failed on TP source" in text
assert not list(
    folder.glob("step*.pt")
), "Failed teacher must not silently update optimizer"
report = dict(
    layout="TP2 PP2 CP2 SP",
    exit_code=result.returncode,
    seconds=seconds,
    bounded_exit=True,
    no_optimizer_update=True,
)
(OUT / "context-failure.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)
