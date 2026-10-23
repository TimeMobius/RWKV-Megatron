"""Run production DP2 accuracy jobs, or all matching-layout resume jobs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare_dp_distill import LAYOUTS, OUT, ROOT, prepare


def main(resume=False):
    prepare(resume)
    statuses = {}
    for dtype in ("bf16", "fp32"):
        for name, (dp, tp, pp, sp) in LAYOUTS.items():
            if dtype == "fp32" and name not in ("dense", "dp2", "dp2-tp2-sp-pp2"):
                continue
            if resume and dp != 2:
                continue
            tag = f"{name}-{dtype}"
            case = tag + ("-resume" if resume else "")
            folder = OUT / case
            folder.mkdir(parents=True, exist_ok=True)
            if not resume and (folder / "step3-rank0.pt").exists():
                print("EXISTING", case, flush=True)
                continue
            env = dict(
                os.environ,
                RWKV_DP_AUDIT_DIR=str(folder),
                RWKV_DP_FP32=str(int(dtype == "fp32")),
            )
            with (OUT / f"{case}.log").open("w") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc-per-node={dp*tp*pp}",
                        str(ROOT / "tests/run_dp_distill_training.py"),
                        "--rwkv-config",
                        str(OUT / f"{case}.yaml"),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            statuses[case] = proc.returncode
            (OUT / f'{"resume" if resume else "precision"}-exitcodes.json').write_text(
                json.dumps(statuses, indent=2) + "\n"
            )
            assert proc.returncode == 0, (case, proc.returncode)
            print("PASS TRAIN", case, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args().resume)
