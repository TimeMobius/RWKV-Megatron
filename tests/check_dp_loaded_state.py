"""Confirm checkpoint loading restores Adam/master state exactly before backward."""

import json

import torch
from compare_dp_distill_resume import exact
from prepare_dp_distill import CKPTS, OUT


def main():
    report = {}
    for tag in ("dp2-tp2-pp2-bf16", "dp2-tp2-sp-pp2-bf16"):
        counts = dict(tensors=0, elements=0)
        for rank in range(8):
            # Megatron rank order for this layout: tp(2), dp(2), pp(2).
            tp, pp = rank % 2, rank // 4
            source = (
                CKPTS
                / tag
                / "iter_0000002"
                / f"mp_rank_{tp:02d}_{pp:03d}/model_optim_rng.pt"
            )
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            loaded = torch.load(
                OUT / f"{tag}-loaded/loaded-optimizer-rank{rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
            exact(checkpoint["optimizer"], loaded, f"{tag}/loaded-rank{rank}", counts)
        report[tag] = dict(**counts, ranks=8, loaded_optimizer_bit_identical=True)
        print("PASS LOADED ADAM/MASTER STATE", tag, counts, flush=True)
    (OUT / "loaded-state.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
