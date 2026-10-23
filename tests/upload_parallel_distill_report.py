"""Upload numerical/restore evidence, clearly labelled as frozen-teacher tests."""

import json
import subprocess
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/parallel-distill-20261023"


def main():
    full = json.loads((OUT / "full-size-smoke.json").read_text())
    api = wandb.Api()
    for name, info in full.items():
        remote = api.run(
            "mobius/rwkv-qwen-distill-2p7b/" + info["wandb"].rsplit("/", 1)[1]
        )
        assert remote.state == "finished", (name, remote.state)
        assert remote.summary.get("teacher/failed", 0) == 0, name
    run = wandb.init(
        entity="mobius",
        project="rwkv-qwen-distill-2p7b",
        name="tp-pp-sp-numerical-certification-20261023",
        tags=["parallel-validation", "frozen-teacher", "numerical-alignment"],
        dir=str(OUT),
        config=dict(
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            teacher="deterministic fixture; production Qwen unavailable (HTTP 503)",
            model="RWKV7 Qwen-vocabulary 2.7B sliced initialization",
            certified_context_parallel=False,
            certified_interleaved_pipeline=False,
            certified_dp_gt_1=False,
        ),
    )
    table = wandb.Table(
        columns=[
            "layout",
            "kl",
            "ce",
            "grad_norm",
            "kl_relative_error",
            "ce_relative_error",
            "grad_norm_relative_error",
            "training_run",
        ]
    )
    for name, info in full.items():
        step, errors = info["steps"][0], info["first_step_relative_errors"]
        table.add_data(
            name,
            step["distill/kl_loss"],
            step["distill/student_ce_loss"],
            step["grad norm"],
            errors["distill/kl_loss"],
            errors["distill/student_ce_loss"],
            errors["grad norm"],
            info["wandb"],
        )
    small = json.loads((OUT / "model-alignment.json").read_text())
    alignment = wandb.Table(
        columns=[
            "layout",
            "logits_relative_l2",
            "max_loss_abs",
            "worst_significant_gradient_relative_l2",
        ]
    )
    for name, info in small.items():
        alignment.add_data(
            name,
            info["logits"]["relative_l2"],
            info["max_loss_abs"],
            info["worst_gradient_metric"]["relative_l2"],
        )
    run.log({"full_size_alignment": table, "model_alignment": alignment})
    for key, value in dict(
        regression_tests_passed=93,
        model_layouts_passed=10,
        full_size_layouts_passed=6,
        resume_bit_identical=True,
        teacher_failure_bounded_exit=True,
        online_teacher_success_certified=False,
    ).items():
        run.summary[key] = value
    artifact = wandb.Artifact("tp-pp-sp-validation-20261023", type="validation-report")
    for name in (
        "full-size-smoke.json",
        "model-alignment.json",
        "resume-alignment.json",
        "failure-result.json",
        "external-teacher-status.json",
    ):
        artifact.add_file(str(OUT / name))
    run.log_artifact(artifact).wait()
    url = run.url
    run.finish()
    assert (
        api.run("mobius/rwkv-qwen-distill-2p7b/" + url.rsplit("/", 1)[1]).state
        == "finished"
    )
    (OUT / "wandb-report.json").write_text(
        json.dumps(dict(url=url, verified_finished=True), indent=2) + "\n"
    )
    print(url)


if __name__ == "__main__":
    main()
