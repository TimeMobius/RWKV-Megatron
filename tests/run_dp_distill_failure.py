"""Inject a teacher error in only DP replica 1 and assert bounded torchrun exit."""

import json
import os
import subprocess
import sys
import time

from prepare_dp_distill import OUT, ROOT


def main():
    start = time.monotonic()
    env = dict(os.environ, RWKV_DP_FAIL="1", RWKV_DP_AUDIT_DIR=str(OUT / "failure"))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        str(ROOT / "tests/run_dp_distill_training.py"),
        "--rwkv-config",
        str(OUT / "dp2-tp2-sp-pp2-bf16.yaml"),
    ]
    with (OUT / "failure.log").open("w") as log:
        proc = subprocess.run(
            ["timeout", "120", *command],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    text = (OUT / "failure.log").read_text()
    assert proc.returncode == 1, proc.returncode
    assert "injected DP1 teacher failure" in text
    report = dict(
        exitcode=proc.returncode,
        seconds=time.monotonic() - start,
        timeout=False,
        injected_replica=1,
        dp=2,
        tp=2,
        pp=2,
        sequence_parallel=True,
    )
    (OUT / "failure.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
