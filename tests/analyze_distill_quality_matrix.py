"""Assert paired samples/checkpoint evaluation and summarize measured GPU costs."""

import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-remaining-20261026"


def ranks(name):
    result = [
        json.loads(path.read_text()) for path in sorted((OUT / name).glob("rank*.json"))
    ]
    assert result, name
    return result


def sample_ids(rows, step):
    values = [
        sample["hash"]
        for row in rows
        if row["tp"] == 0 and row["pp"] == 0
        for batch in row["batches"]
        if batch["step"] == step
        for sample in batch["samples"]
    ]
    return Counter(values)


def main():
    runs = json.loads((OUT / "quality-runs.json").read_text())
    reference = ranks("dense")
    ref_ce = runs["dense"]["validation_ce"]
    assert ref_ce is not None
    report = {}
    for name in ("dense", "tp2", "tp2sp", "pp2", "combined"):
        rows = ranks(name)
        layout = runs[name]["layout"]
        world = layout["dp"] * layout["tp"] * layout["pp"]
        assert len(rows) == world
        for step in range(1, 9):
            assert sample_ids(rows, step) == sample_ids(reference, step), (
                name,
                step,
                "train samples",
            )
            assert sum(sample_ids(rows, step).values()) == 8
            assert all(value == 1 for value in sample_ids(rows, step).values()), (
                name,
                step,
                "duplicates",
            )
        assert sample_ids(rows, None) == sample_ids(reference, None), (
            name,
            "validation samples",
        )
        assert all(row["teacher_requests"] == 0 for row in rows), (
            name,
            "HTTP request during replay",
        )
        steps = {m["step"] for row in rows for m in row["measurements"]}
        assert steps == set(range(1, 9))
        timings = [
            max(
                m["seconds"]
                for row in rows
                for m in row["measurements"]
                if m["step"] == step
            )
            for step in range(3, 9)
        ]
        mean = statistics.mean(timings)
        ce = runs[name]["validation_ce"]
        ppl = runs[name]["validation_ppl"]
        assert math.isfinite(ce) and math.isfinite(ppl)
        assert abs(ce / ref_ce - 1) < 0.005, (name, ce, ref_ce)
        assert abs(ppl / runs["dense"]["validation_ppl"] - 1) < 0.01, (name, ppl)
        assert all(
            m["skipped"] == 0 and math.isfinite(m["grad_norm"])
            for row in rows
            for m in row["measurements"]
        )
        valid_tokens = sum(
            sample["valid_tokens"]
            for row in rows
            if row["tp"] == 0 and row["pp"] == 0
            for batch in row["batches"]
            if batch["step"] in range(3, 9)
            for sample in batch["samples"]
        )
        report[name] = dict(
            layout=layout,
            validation_ce=ce,
            validation_ppl=ppl,
            validation_ce_relative_error=abs(ce / ref_ce - 1),
            validation_ppl_relative_error=abs(
                ppl / runs["dense"]["validation_ppl"] - 1
            ),
            global_samples_identical=True,
            no_http_during_replay=True,
            steady_steps=list(range(3, 9)),
            mean_step_seconds=mean,
            padded_tokens_per_second=8 * 512 / mean,
            useful_tokens_per_second=valid_tokens / sum(timings),
            padded_tokens_per_second_per_gpu=8 * 512 / (mean * world),
            peak_allocated_mib=max(
                m["max_allocated_mib"] for row in rows for m in row["measurements"]
            ),
            peak_reserved_mib=max(
                m["max_reserved_mib"] for row in rows for m in row["measurements"]
            ),
            wandb=runs[name]["wandb"],
        )
    for name in ("dense", "combined"):
        loaded = name + "-reload"
        assert runs[loaded]["load"] == runs[name]["save"]
        assert (
            abs(runs[loaded]["validation_ce"] - runs[name]["validation_ce"]) < 1e-5
        ), (loaded, "checkpoint CE")
        # A training run evaluates validation at its interval and at exit;
        # eval-only reload evaluates it once. Compare the actual sample set.
        assert set(sample_ids(ranks(loaded), None)) == set(sample_ids(reference, None))
        report[name]["checkpoint_reload_ce"] = runs[loaded]["validation_ce"]
        report[name]["checkpoint_reload_verified"] = True
    report["initial"] = runs["initial"]
    report["comparison_scope"] = (
        "2.7B, 8 optimizer steps / 64 same training samples, 32 fixed validation samples per eval, seq512, identical recorded real-Qwen top100; cache replay GPU cost, excludes live HTTP throughput; no long-run quality claim"
    )
    (OUT / "quality-performance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
