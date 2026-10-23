"""Check DP sample partition, production reductions and per-parameter alignment."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from prepare_dp_distill import LAYOUTS, OUT, ROOT

sys.path.insert(0, str(ROOT / "training-smoke/audit"))
from distributed import metric

from rwkv_conversion import convert_pipeline_stage


def load_case(tag, step):
    records = [
        torch.load(path, weights_only=False)
        for path in sorted((OUT / tag).glob(f"step{step}-rank*.pt"))
    ]
    assert records, (tag, step)
    return records


def canonical(records, field, tp, pp, dp=0):
    selected = [r for r in records if r["dp"] == dp]
    cfg = SimpleNamespace(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )
    merged = {}
    for stage in range(pp):
        shards = sorted(
            [r for r in selected if r["pp"] == stage], key=lambda r: r["tp"]
        )
        merged.update(convert_pipeline_stage([r[field] for r in shards], stage, cfg, 0))
    return merged


def replicas(records, dp, tp, pp):
    assert len(records) == dp * tp * pp
    for stage in range(pp):
        for shard in range(tp):
            copies = sorted(
                [r for r in records if r["pp"] == stage and r["tp"] == shard],
                key=lambda r: r["dp"],
            )
            for other in copies[1:]:
                for field in ("grads", "before", "after"):
                    assert copies[0][field].keys() == other[field].keys()
                    for key, value in copies[0][field].items():
                        assert torch.equal(value, other[field][key]), (
                            field,
                            stage,
                            shard,
                            key,
                        )


def samples(records, dp, pp):
    sets = []
    logits = {}
    for replica in range(dp):
        ends = [r for r in records if r["dp"] == replica]
        reference = ends[0]["samples"]
        assert len(reference) == len(set(reference)) == 16 // dp
        assert all(
            r["samples"] == reference for r in ends
        ), "endpoint or TP samples differ"
        sets.append(set(reference))
        final = next(r for r in ends if r["pp"] == pp - 1 and r["tp"] == 0)
        assert set(final["logits"]) == sets[-1]
        logits.update(final["logits"])
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            assert not a & b, "DP samples overlap"
    return set.union(*sets), logits


def compare(dtypes=("bf16", "fp32")):
    report = {}
    for dtype in dtypes:
        reference = load_case("dense-" + dtype, 1)
        expected_samples, ref_logits = samples(reference, 1, 1)
        expected_grads = canonical(reference, "grads", 1, 1)
        expected_after = canonical(reference, "after", 1, 1)
        expected_loss = reference[0]["losses"]
        for name, (dp, tp, pp, sp) in LAYOUTS.items():
            if dp != 2 or (dtype == "fp32" and name not in ("dp2", "dp2-tp2-sp-pp2")):
                continue
            tag = name + "-" + dtype
            for step in (1, 2, 3):
                records = load_case(tag, step)
                replicas(records, dp, tp, pp)
                ref_set, _ = samples(load_case("dense-" + dtype, step), 1, 1)
                actual_set, _ = samples(records, dp, pp)
                assert actual_set == ref_set, (tag, step, "global samples")
                assert all(r["skipped"] == 0 for r in records)
            records = load_case(tag, 1)
            actual_samples, actual_logits = samples(records, dp, pp)
            assert actual_samples == expected_samples
            keys = sorted(expected_samples)
            logit_diff = metric(
                torch.stack([actual_logits[k] for k in keys]),
                torch.stack([ref_logits[k] for k in keys]),
            )
            assert logit_diff["relative_l2"] < (1e-5 if dtype == "fp32" else 0.02), (
                tag,
                logit_diff,
            )
            grads = canonical(records, "grads", tp, pp)
            assert grads.keys() == expected_grads.keys()
            grad_metrics = {
                key: metric(value, expected_grads[key]) for key, value in grads.items()
            }
            assert all(m["finite"] for m in grad_metrics.values())
            significant = {
                key: value
                for key, value in grad_metrics.items()
                if value["reference_l2"] >= 1e-3
            }
            worst = max(significant, key=lambda k: significant[k]["relative_l2"])
            assert significant[worst]["relative_l2"] < (
                1e-4 if dtype == "fp32" else 0.05
            ), (tag, worst, significant[worst])
            loss_record = next(
                r
                for r in records
                if r["pp"] == pp - 1 and r["tp"] == 0 and r["dp"] == 0
            )
            errors = {
                key: abs(loss_record["losses"][key] - expected_loss[key])
                / abs(expected_loss[key])
                for key in ("lm loss", "distill/kl_loss", "distill/student_ce_loss")
            }
            assert max(errors.values()) < (1e-5 if dtype == "fp32" else 0.005), (
                tag,
                errors,
            )
            after = canonical(records, "after", tp, pp)
            update_metrics = {
                key: metric(value, expected_after[key]) for key, value in after.items()
            }
            report[tag] = dict(
                logits=logit_diff,
                loss_relative_errors=errors,
                worst_gradient=worst,
                worst_gradient_metric=significant[worst],
                gradients=grad_metrics,
                max_after_parameter_relative_l2=max(
                    m["relative_l2"] for m in update_metrics.values()
                ),
                dp_replicas_bit_identical=True,
                global_samples_match=True,
                dp_samples_disjoint=True,
                steps_checked=3,
            )
            print(
                "PASS DP ALIGNMENT",
                tag,
                logit_diff["relative_l2"],
                significant[worst]["relative_l2"],
                flush=True,
            )
    (OUT / "precision.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["bf16", "fp32"])
    compare(parser.parse_args().dtypes)
