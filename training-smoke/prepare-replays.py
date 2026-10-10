#!/usr/bin/env python3
"""Recreate frozen baseline sources without switching active repository branches."""

import io
import shutil
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "training-smoke/baseline-replay"


def extract(repo, revision, destination, *paths):
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", revision, *paths])
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, filter="data")


def link(path, target):
    if not path.is_symlink() and not path.exists():
        path.symlink_to(target)


def main():
    if BASELINE.exists():
        raise SystemExit(f"Snapshot already exists; preserve it: {BASELINE}")
    extract(ROOT, "34e447cdc186a2763ae3e56f3b776772be8f719c", BASELINE,
            "pretrain_rwkv7.py", "rwkv_megatron", "cuda")
    extract(ROOT / "3rdparty/Megatron-LM-0.11.0",
            "c11ea0b3cd8121a289b27985fdcd402322444996", BASELINE / "meg-source", "megatron")
    extract(ROOT / "3rdparty/flash-linear-attention-260115",
            "3cf180339b8a1cbad823f553541cd531d18670ea", BASELINE / "fla-source", "fla")
    link(BASELINE / "megatron", "meg-source/megatron")
    link(BASELINE / "fla", "fla-source/fla")
    fla_only = ROOT / "training-smoke/fla-only-replay"
    fla_only.mkdir(exist_ok=True)
    shutil.copy2(BASELINE / "pretrain_rwkv7.py", fla_only / "pretrain_rwkv7.py")
    for name in ("rwkv_megatron", "cuda", "megatron"):
        link(fla_only / name, "../baseline-replay/" + name)
    link(fla_only / "fla", "../../3rdparty/flash-linear-attention-260115/fla")


if __name__ == "__main__":
    main()
