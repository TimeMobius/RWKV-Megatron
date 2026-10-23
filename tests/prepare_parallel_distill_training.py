"""Create repeatable configs for training/resume and frozen full-size smoke.

Run the model audit first to generate cropped-native.pth. --resume exposes
iteration 2 through a separate tracker without changing the original checkpoint.
"""

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/parallel-distill-20261023"
CKPTS = ROOT / "training-smoke/checkpoints/parallel-distill-20261023"


def prepare(resume=False):
    OUT.mkdir(exist_ok=True, parents=True)
    base = yaml.safe_load(
        (
            ROOT / "configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-single.yaml"
        ).read_text()
    )
    base["language_model"].update(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        ffn_hidden_size=1024,
        D_AAA_LORA=16,
        D_DECAY_LORA=16,
        D_GATE_LORA=16,
        D_MV_LORA=16,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
    )
    base.update(
        tokenizer_type="NullTokenizer",
        vocab_size=119,
        tokenizer_model=None,
        make_vocab_size_divisible_by=16,
        load_pretrained_rwkv=str(ROOT / "training-smoke/audit/cropped-native.pth"),
        data_path=None,
        mock_data=True,
        split="80,10,10",
        data_cache_path=str(OUT / "cache"),
        num_workers=0,
        micro_batch_size=2,
        global_batch_size=8,
        eval_micro_batch_size=2,
        eval_global_batch_size=8,
        train_iters=3,
        lr_decay_iters=3,
        eval_interval=1,
        eval_iters=1,
        lr=1e-4,
        min_lr=1e-4,
        rwkv_distill_valid_vocab_size=119,
        rwkv_distill_ce_weight=0.3,
        rwkv_teacher_async_prefetch=False,
        wandb_project=None,
        wandb_exp_name=None,
        wandb_save_dir=str(OUT / "wandb"),
        tensorboard_dir=None,
        save_interval=2,
        distributed_timeout_minutes=3,
        log_throughput=False,
    )
    for name, tp, pp, sp in [
        ("tp2", 2, 1, False),
        ("tp2-sp", 2, 1, True),
        ("pp2", 1, 2, False),
        ("combined", 2, 2, True),
    ]:
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["model_parallel"].update(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            sequence_parallel=sp,
        )
        cfg["save"] = str(CKPTS / name)
        (OUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if name == "combined" and resume:
            source = CKPTS / "resume-source"
            source.mkdir(exist_ok=True, parents=True)
            target = CKPTS / "combined/iter_0000002"
            assert target.is_dir(), "Run combined training before --resume"
            link = source / "iter_0000002"
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
            (source / "latest_checkpointed_iteration.txt").write_text("2")
            cfg.update(load=str(source), save=str(CKPTS / "combined-resume"))
            (OUT / "combined-resume.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False)
            )
    for name in ("dense", "tp2", "tp2-sp", "pp2", "tp2-sp-pp2", "tp4-sp-pp2"):
        cfg = yaml.safe_load(
            (ROOT / f"configs/config-qwen-2p7b-distill-{name}-smoke.yaml").read_text()
        )
        folder = OUT / f"frozen-{name}"
        cfg.update(
            wandb_exp_name=f"qwen-2p7b-frozen-parallel-{name}-20261023",
            wandb_save_dir=str(folder / "wandb"),
            tensorboard_dir=str(folder / "tensorboard"),
        )
        (OUT / f"frozen-{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    prepare(parser.parse_args().resume)
