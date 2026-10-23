"""Repeat the two BF16 combined layouts with Megatron deterministic_mode.

Keep original non-deterministic runs intact, including the observed Adam
rounding differences; strict resume assertions are never changed to allclose.
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml
from prepare_dp_distill import CKPTS, OUT, ROOT


def main():
    for name in ("dp2-tp2-pp2-bf16", "dp2-tp2-sp-pp2-bf16"):
        tag = name + "-deterministic"
        cfg = yaml.safe_load((OUT / f"{name}.yaml").read_text())
        cfg.update(deterministic_mode=True, save=str(CKPTS / tag), load=None)
        for resume in (False, True):
            case = tag + ("-resume" if resume else "")
            if resume:
                source = CKPTS / (tag + "-resume-source")
                source.mkdir(parents=True, exist_ok=True)
                link = source / "iter_0000002"
                if not link.exists():
                    link.symlink_to(
                        CKPTS / tag / "iter_0000002", target_is_directory=True
                    )
                (source / "latest_checkpointed_iteration.txt").write_text("2")
                cfg.update(load=str(source), save=str(CKPTS / (tag + "-resumed")))
            config = OUT / f"{case}.yaml"
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            env = dict(os.environ, RWKV_DP_AUDIT_DIR=str(OUT / case))
            with (OUT / f"{case}.log").open("w") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc-per-node=8",
                        str(ROOT / "tests/run_dp_distill_training.py"),
                        "--rwkv-config",
                        str(config),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            assert proc.returncode == 0, (case, proc.returncode)
            print("PASS DETERMINISTIC TRAIN", case, flush=True)


if __name__ == "__main__":
    main()
