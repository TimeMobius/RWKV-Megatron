"""Prepare production-training DP2 configs with a fixed global batch."""

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/dp2-distill-20261023"
CKPTS = ROOT / "training-smoke/checkpoints/dp2-distill-20261023"
LAYOUTS = {
    "dense": (1, 1, 1, False),
    "dp2": (2, 1, 1, False),
    "dp2-tp2": (2, 2, 1, False),
    "dp2-tp2-sp": (2, 2, 1, True),
    "dp2-pp2": (2, 1, 2, False),
    "dp2-tp2-pp2": (2, 2, 2, False),
    "dp2-tp2-sp-pp2": (2, 2, 2, True),
}


def prepare(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    from prepare_parallel_distill_training import prepare as prepare_base

    prepare_base()
    base = yaml.safe_load(
        (ROOT / "training-smoke/parallel-distill-20261023/combined.yaml").read_text()
    )
    base.update(
        global_batch_size=16,
        eval_global_batch_size=16,
        save_interval=2,
        train_iters=3,
        lr_decay_iters=3,
        load=None,
        eval_interval=3,
        rwkv_key_fusion=False,
        make_vocab_size_divisible_by=16,
        lr=1e-4,
        min_lr=1e-4,
        data_cache_path=str(OUT / "cache"),
    )
    for dtype in ("bf16", "fp32"):
        for name, (dp, tp, pp, sp) in LAYOUTS.items():
            if dtype == "fp32" and name not in ("dense", "dp2", "dp2-tp2-sp-pp2"):
                continue
            tag = f"{name}-{dtype}"
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["model_parallel"].update(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                sequence_parallel=sp,
                bf16=dtype == "bf16",
                gradient_accumulation_fusion=False,
            )
            cfg["save"] = str(CKPTS / tag)
            (OUT / f"{tag}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            if resume and dp == 2:
                source = CKPTS / f"{tag}-resume-source"
                source.mkdir(parents=True, exist_ok=True)
                target = CKPTS / tag / "iter_0000002"
                assert target.is_dir(), target
                link = source / "iter_0000002"
                if not link.exists():
                    link.symlink_to(target, target_is_directory=True)
                (source / "latest_checkpointed_iteration.txt").write_text("2")
                cfg.update(load=str(source), save=str(CKPTS / f"{tag}-resumed"))
                (OUT / f"{tag}-resume.yaml").write_text(
                    yaml.safe_dump(cfg, sort_keys=False)
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    prepare(parser.parse_args().resume)
