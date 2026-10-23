"""Publish causal precision controls and strict replay evidence, including failures."""

import json
import subprocess
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-numerics-20261023"
HISTORY = ROOT / "training-smoke/dp2-distill-20261023"


def main():
    precision = json.loads((OUT / "precision-diagnosis.json").read_text())
    replays = json.loads((OUT / "replay-controls.json").read_text())
    historical = json.loads((HISTORY / "resume.json").read_text())
    loaded = json.loads((HISTORY / "loaded-state.json").read_text())
    accumulation = json.loads((OUT / "accumulation-diagnosis.json").read_text())
    warm = json.loads((OUT / "warm-runtime-control.json").read_text())
    assert all(v["loaded_optimizer_bit_identical"] for v in loaded.values())
    assert len(accumulation) == 3 and all(
        v["per_micro_accumulation_exact"] and v["captured_final_reduction_replay_exact"]
        for v in accumulation.values()
    )
    assert len(warm) == 2 and all(v["bit_identical"] for v in warm.values())
    assert precision["vjp_reference_matches_actual"]["max_abs"] == 0
    assert precision["first_row_replay_matches_tp"]["max_abs"] == 0
    assert precision["tp2_vs_dp2_tp2_w1"]["max_abs"] == 0
    for case in ("tp2", "tp2-row32", "tp2-sp-none", "tp2-sp-row32-none"):
        assert case in precision
    for case in (
        "combined-original-none",
        "combined-sp-original-none",
        "combined-cold",
        "combined-sp-cold",
    ):
        assert case in replays and "bit_identical" in replays[case]
    run = wandb.init(
        entity="mobius",
        project="rwkv-qwen-distill-2p7b",
        name="tp-bf16-numerical-cause-and-replay-20261023",
        dir=str(OUT),
        tags=[
            "numerical-diagnosis",
            "fixed-teacher",
            "tp-rounding",
            "checkpoint-recovery",
        ],
        config=dict(
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            teacher="deterministic token-indexed fixture, no HTTP",
            model="4 layers, hidden256 cropped native weights",
            global_batch=16,
            micro_batch=2,
            production_precision_changed=False,
            row32="diagnostic monkeypatch only: FP32 row GEMM outputs and TP reduction",
        ),
    )
    table = wandb.Table(
        columns=[
            "case",
            "logits_relative_l2",
            "layer1_w1_relative_l2",
            "worst_significant_gradient_relative_l2",
            "worst_parameter",
        ]
    )
    for case, info in precision.items():
        if "worst_significant" in info:
            table.add_data(
                case,
                info["logits"]["relative_l2"],
                info["w1"]["relative_l2"],
                info["worst_significant"]["relative_l2"],
                info["worst_significant_parameter"],
            )
    vjp = wandb.Table(
        columns=["fixed_input_upstream_control", "w1_relative_l2", "w1_max_abs"]
    )
    for case, info in precision["vjp_swap"].items():
        vjp.add_data(case, info["relative_l2"], info["max_abs"])
    restore = wandb.Table(
        columns=[
            "case",
            "comparison_type",
            "snapshot_differences",
            "checkpoint_differences",
            "strict_bit_identical",
            "ranks_with_kernel_metadata_on_both_sides",
            "captured_kernel_config_differences",
        ]
    )
    for case, info in replays.items():
        restore.add_data(
            case,
            info["comparison_type"],
            sum(len(v) for v in info["ranks"].values()),
            len(info["checkpoint_differences"]),
            info["bit_identical"],
            len(info["kernel_config_compared_ranks"]),
            len(info["kernel_config_differences"]),
        )
    run.log(
        dict(precision_controls=table, fixed_vjp_controls=vjp, replay_controls=restore)
    )
    run.summary.update(
        dict(
            row_rounding_replay_bit_identical=True,
            dense_vjp_replay_bit_identical=True,
            dp1_tp2_vs_dp2_tp2_w1_identical=True,
            loaded_adam_and_master_state_bit_identical=True,
            original_strict_restore_failures=sum(
                not v["bit_identical"] for v in historical.values()
            ),
            original_strict_restore_issue_resolved=False,
            original_root_operator_identified=False,
            local_microgradient_and_dp_replay_bit_identical=True,
            direct_vs_rewarmed_runtime_bit_identical=True,
            production_precision_changed=False,
            new_replay_cases=len(replays),
            new_replay_strict_passed=sum(v["bit_identical"] for v in replays.values()),
        )
    )
    artifact = wandb.Artifact(
        "distill-bf16-numerical-diagnosis-20261023", type="validation-report"
    )
    for filename in (
        "precision-diagnosis.json",
        "replay-diagnosis.json",
        "replay-controls.json",
        "accumulation-diagnosis.json",
        "warm-runtime-control.json",
    ):
        artifact.add_file(str(OUT / filename))
    for filename in (
        "resume.json",
        "default-resume-rounding.json",
        "loaded-state.json",
    ):
        artifact.add_file(str(HISTORY / filename), name="historical/" + filename)
    artifact.add_file(str(ROOT / "docs/distillation-numerics-20261023.md"))
    run.log_artifact(artifact).wait()
    url = run.url
    run.finish()
    assert (
        wandb.Api().run("mobius/rwkv-qwen-distill-2p7b/" + url.rsplit("/", 1)[1]).state
        == "finished"
    )
    (OUT / "wandb.json").write_text(
        json.dumps(dict(url=url, verified_finished=True), indent=2) + "\n"
    )
    print(url)


if __name__ == "__main__":
    main()
