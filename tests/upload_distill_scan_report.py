"""Publish the scan-layout causal proof and production deterministic replay checks."""

import json
import subprocess
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-numerics-20261023"


def main():
    reconstruction = json.loads(
        (OUT / "dgk-gradient-reconstruction-20261026.json").read_text()
    )
    replays = json.loads((OUT / "replay-controls.json").read_text())
    deterministic = json.loads((OUT / "deterministic-replay-20261026.json").read_text())
    precision = json.loads((OUT / "deterministic-precision-20261026.json").read_text())
    assert len(deterministic) == 2 and all(
        v["bit_identical"] for v in deterministic.values()
    )
    assert precision.keys() == deterministic.keys()
    assert all(
        v["logits"]["relative_l2"] < 0.02
        and v["worst_significant_gradient"]["relative_l2"] < 0.05
        for v in precision.values()
    )
    for sp in (False, True):
        prefix = "combined-" + ("sp-" if sp else "") + "original-"
        evidence = reconstruction[prefix + "dgktrace-20261026"]
        matches = [v for v in evidence["controls"] if v["historical_unequal"] == 0]
        assert len(matches) == 1 and matches[0]["rank4_to_7_pattern"] == [0, 1, 0, 0]
        assert replays[prefix + "dgk-historical-20261026"]["bit_identical"]
        assert not replays[prefix + "dgk-fixed-20261026"]["bit_identical"]
    run = wandb.init(
        entity="mobius",
        project="rwkv-qwen-distill-2p7b",
        name="bf16-replay-reverse-scan-cause-and-fix-20261026",
        dir=str(OUT),
        tags=[
            "checkpoint-recovery",
            "root-cause",
            "fixed-teacher",
            "deterministic-mode",
        ],
        config=dict(
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            teacher="fixed fixture, no HTTP",
            layers=4,
            hidden_size=256,
            seq_length=128,
            head_dim=64,
            global_batch=16,
            micro_batch=2,
            world_size=8,
            default_mode_changed=False,
            deterministic_scan_config="BK32/warps4/stages2",
            historical_rank5_config_override="diagnostic proof only, never production",
        ),
    )
    table = wandb.Table(
        columns=[
            "case",
            "comparison_type",
            "strict_bit_identical",
            "snapshot_differences",
            "checkpoint_differences",
            "kernel_config_differences",
        ]
    )
    selected = {name: info for name, info in replays.items() if "20261026" in name}
    for name, info in selected.items():
        table.add_data(
            name,
            info["comparison_type"],
            info["bit_identical"],
            sum(len(v) for v in info["ranks"].values()),
            len(info["checkpoint_differences"]),
            len(info["kernel_config_differences"]),
        )
    reconstruction_table = wandb.Table(
        columns=[
            "case",
            "rank4_BK64",
            "rank5_BK64",
            "rank6_BK64",
            "rank7_BK64",
            "historical_unequal_gradient_elements",
        ]
    )
    for name, evidence in reconstruction.items():
        for info in evidence["controls"]:
            reconstruction_table.add_data(
                name, *info["rank4_to_7_pattern"], info["historical_unequal"]
            )
    run.log(
        dict(strict_replay=table, fixed_scan_vjp_reconstruction=reconstruction_table)
    )
    run.summary.update(
        dict(
            root_operator_identified=True,
            root_operator="chunk_dplr_bwd_dgk_kernel",
            root_cause="autotuned BK/warp layout changes FP32 reverse-scan addition tree before BF16 rounding",
            original_default_failure_causally_reconstructed_cases=2,
            historical_match_requires_recorded_layout=True,
            deterministic_mode_fixed_scan_cases=2,
            deterministic_mode_strict_replay_passed=True,
            independent_autotuning_for_other_kernels=True,
            default_mode_bitwise_guarantee=False,
            adam_formula_changed=False,
            accumulation_dtype_changed=False,
            student_quality_long_run_validated=False,
        )
    )
    artifact = wandb.Artifact(
        "bf16-replay-scan-cause-and-fix-20261026", type="validation-report"
    )
    for name in (
        "decay-vjp-20261026.json",
        "dgk-kernel-controls-20261026.json",
        "dgk-gradient-reconstruction-20261026.json",
        "deterministic-replay-20261026.json",
        "deterministic-precision-20261026.json",
        "replay-controls.json",
    ):
        artifact.add_file(str(OUT / name))
    artifact.add_file(str(ROOT / "docs/distillation-replay-20261026.md"))
    for case in deterministic:
        for suffix in ("", "-resume"):
            for rank in range(8):
                artifact.add_file(
                    str(OUT / (case + suffix) / f"kernels-step3-rank{rank}.json"),
                    name=f"kernel-configs/{case+suffix}/rank{rank}.json",
                )
    run.log_artifact(artifact).wait()
    url = run.url
    run.finish()
    assert (
        wandb.Api().run("mobius/rwkv-qwen-distill-2p7b/" + url.rsplit("/", 1)[1]).state
        == "finished"
    )
    (OUT / "wandb-scan-20261026.json").write_text(
        json.dumps(dict(url=url, verified_finished=True), indent=2) + "\n"
    )
    print(url)


if __name__ == "__main__":
    main()
