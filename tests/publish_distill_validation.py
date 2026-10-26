"""Upload compact validation evidence and tables to the existing W&B project."""

import json
import subprocess
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-remaining-20261026"
reports = {}
for name in (
    "quality-runs",
    "quality-performance",
    "optimizer-dp4",
    "context",
    "context-failure",
    "context-online",
):
    path = OUT / (name + ".json")
    if path.exists():
        reports[name] = json.loads(path.read_text())
run = wandb.init(
    entity="mobius",
    project="rwkv-qwen-distill-2p7b",
    name="parallel-validation-summary-20261026",
    mode="online",
    dir=str(OUT / "report-wandb"),
    tags=["distill", "validation", "parallel", "checkpoint"],
    config=dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        model="2.7B for quality/performance; small fixture for optimizer/CP",
        twelve_hour_longrun="pending, must run last",
    ),
)
artifact = wandb.Artifact(
    "parallel-validation-20261026",
    type="validation-evidence",
    description="Fixed Qwen targets, quality, GPU cost, optimizer and CP/restart evidence; no longrun claim.",
)
for name in reports:
    artifact.add_file(str(OUT / (name + ".json")))
for name in ("docs/distillation-remaining-20261026.md", "todolist.txt"):
    artifact.add_file(str(ROOT / name))
run.log_artifact(artifact)
for name in ("optimizer-dp4", "context"):
    if name not in reports:
        continue
    table = wandb.Table(
        columns=[
            "case",
            "layout",
            "gradient_relative_l2",
            "logits_relative_l2",
            "strict_resume",
            "peak_allocated_mib",
        ]
    )
    for key, row in reports[name].items():
        assert row.get("strict_resume_bit_identical") is True, (name, key)
        table.add_data(
            key,
            json.dumps(row["layout"]),
            row["worst_gradient"]["relative_l2"],
            row.get("logits", {}).get("relative_l2"),
            True,
            row.get("peak_allocated_mib"),
        )
    run.log({name: table})
run.summary["twelve_hour_longrun_completed"] = False
run.summary["quality_performance_report"] = reports.get("quality-performance")
url, runid = run.url, run.id
run.finish()
remote = wandb.Api().run("mobius/rwkv-qwen-distill-2p7b/" + runid)
assert remote.state == "finished", remote.state
(OUT / "wandb-summary.json").write_text(
    json.dumps(dict(url=url, verified_finished=True), indent=2) + "\n"
)
print(url)
