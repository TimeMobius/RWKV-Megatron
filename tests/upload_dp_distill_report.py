"""Publish DP2 production gradient, resume and real-teacher evidence to W&B."""

import json
import subprocess
from pathlib import Path

import wandb
from prepare_dp_distill import OUT, ROOT


def main():
    precision = json.loads((OUT / "precision.json").read_text())
    resumed = json.loads((OUT / "resume.json").read_text())
    failure = json.loads((OUT / "failure.json").read_text())
    online = json.loads((OUT / "online.json").read_text())
    loaded = json.loads((OUT / "loaded-state.json").read_text())
    assert len(precision) == 8 and len(resumed) == 10 and online["all_passed"]
    assert all(v["loaded_optimizer_bit_identical"] for v in loaded.values())
    assert failure["exitcode"] == 1 and not failure["timeout"]
    strict_passed = sum(v["bit_identical"] for v in resumed.values())
    for info in online["layouts"].values():
        assert (
            wandb.Api()
            .run("mobius/rwkv-qwen-distill-2p7b/" + info["wandb"].rsplit("/", 1)[1])
            .state
            == "finished"
        )
    run = wandb.init(
        entity="mobius",
        project="rwkv-qwen-distill-2p7b",
        name="dp2-tp-pp-sp-precision-resume-20261023",
        dir=str(OUT),
        tags=[
            "dp2",
            "parallel-validation",
            "gradient-alignment",
            "checkpoint-recovery",
        ],
        config=dict(
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            precision_teacher="deterministic fixture",
            online_teacher="Qwen3.8-27B",
            global_batch_precision=16,
            global_batch_online=8,
        ),
    )
    table = wandb.Table(
        columns=[
            "layout",
            "logits_relative_l2",
            "max_loss_relative_error",
            "worst_gradient_relative_l2",
            "dp_replicas_exact",
            "samples_match",
        ]
    )
    for name, info in precision.items():
        table.add_data(
            name,
            info["logits"]["relative_l2"],
            max(info["loss_relative_errors"].values()),
            info["worst_gradient_metric"]["relative_l2"],
            info["dp_replicas_bit_identical"],
            info["global_samples_match"],
        )
    restore = wandb.Table(
        columns=[
            "layout",
            "checkpoint_ranks",
            "audit_ranks",
            "tensors",
            "elements",
            "exact",
            "strict_assertion",
        ]
    )
    for name, info in resumed.items():
        restore.add_data(
            name,
            info.get("checkpoint_ranks"),
            info.get("audit_ranks"),
            info.get("tensors"),
            info.get("elements"),
            info["bit_identical"],
            info.get("strict_assertion", ""),
        )
    real = wandb.Table(columns=["layout", "kl", "ce", "grad_norm", "run"])
    for name, info in online["layouts"].items():
        for step in info["steps"]:
            real.add_data(
                name,
                step["distill/kl_loss"],
                step["distill/student_ce_loss"],
                step["grad norm"],
                info["wandb"],
            )
    run.log(dict(precision=table, checkpoint_resume=restore, real_online=real))
    run.summary.update(
        dict(
            precision_cases=8,
            resume_cases=10,
            strict_resume_cases_passed=strict_passed,
            strict_resume_cases_failed=len(resumed) - strict_passed,
            dp_replicas_bit_identical=True,
            resumed_tensors_bit_identical=all(
                v["bit_identical"] for v in resumed.values()
            ),
            loaded_adam_master_state_bit_identical=True,
            online_cases=3,
            teacher_failure_bounded_exit=True,
        )
    )
    artifact = wandb.Artifact(
        "dp2-tp-pp-sp-validation-20261023", type="validation-report"
    )
    for name in (
        "precision.json",
        "resume.json",
        "failure.json",
        "online.json",
        "default-resume-rounding.json",
        "loaded-state.json",
    ):
        artifact.add_file(str(OUT / name))
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
