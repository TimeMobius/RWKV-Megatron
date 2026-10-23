"""Summarize layer/VJP traces and identify the earliest replay divergence."""

import json
import sys
from pathlib import Path

import torch
from compare_dp_distill import canonical
from einops import einsum

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training-smoke/audit"))
from distributed import metric

OUT = ROOT / "training-smoke/distill-numerics-20261023"


def records(name, step=1, trace=False):
    return [
        torch.load(path, weights_only=False)
        for path in sorted(
            (OUT / name).glob(f'{"trace-" if trace else ""}step{step}-rank*.pt')
        )
    ]


def merged_modules(name):
    rows = sorted(
        [r for r in records(name, trace=True) if r["dp"] == 0], key=lambda r: r["tp"]
    )
    result = {}
    for key, mods in rows[0]["modules"].items():
        result[key] = {}
        for name, entry in mods.items():
            result[key][name] = {}
            for field in entry:
                values = [r["modules"][key][name][field] for r in rows]
                is_input = field.startswith("input")
                sharded = (
                    ("core_attention" in name)
                    or ("linear_proj" in name and is_input)
                    or ("linear_down" in name and is_input)
                    or ("linear_key" in name and not is_input)
                    or ("bottleneck_decay" in name and not is_input)
                )
                result[key][name][field] = (
                    torch.cat(values, -1) if sharded else values[0]
                )
    return result


def precision():
    reference = records("dense")
    ref = canonical(reference, "grads", 1, 1)
    report = {}
    for case, tp in [("tp2", 2), ("dp2-tp2", 2), ("dense-row32", 1), ("tp2-row32", 2)]:
        rows = records(case)
        if not rows:
            continue
        grad = canonical(rows, "grads", tp, 1)
        metrics = {key: metric(value, ref[key]) for key, value in grad.items()}
        k = "blocks.1.att.w1"
        ref_logits = reference[0]["logits"]
        actual_logits = {}
        for r in rows:
            if r["tp"] == 0:
                actual_logits.update(r["logits"])
        logit_keys = sorted(ref_logits)
        logits = metric(
            torch.stack([actual_logits[k] for k in logit_keys]),
            torch.stack([ref_logits[k] for k in logit_keys]),
        )
        report[case] = dict(
            logits=logits,
            w1=metrics[k],
            worst_significant=max(
                (v for v in metrics.values() if v["reference_l2"] >= 1e-3),
                key=lambda v: v["relative_l2"],
            ),
        )
    if records("dense-row32") and records("tp2-row32"):
        a = canonical(records("dense-row32"), "grads", 1, 1)
        b = canonical(records("tp2-row32"), "grads", 2, 1)
        report["row32-vs-row32"] = {
            k: metric(b[k], a[k]) for k in ("blocks.1.att.w1", "blocks.1.att.w0")
        }
    a = merged_modules("dense")
    b = merged_modules("tp2")
    keys = sorted(a.keys())
    layer_metrics = {}
    for name, entry in a[keys[0]].items():
        layer_metrics[name] = {
            field: metric(
                torch.stack([b[k][name][field] for k in keys]),
                torch.stack([a[k][name][field] for k in keys]),
            )
            for field in entry
        }
    report["layer_metrics"] = layer_metrics
    # Re-run the suspect bottleneck VJP with fixed inputs and incoming gradients.
    weights = reference[0]["before"]
    w1 = weights["decoder.layers.1.self_attention.bottleneck_decay.weight_down"].cuda()
    w2 = weights["decoder.layers.1.self_attention.bottleneck_decay.weight_up"].cuda()
    target = "layer1.self_attention.bottleneck_decay"
    replay = {}
    for label, xsource, gsource in [
        ("dense", a, a),
        ("x_tp_g_dense", b, a),
        ("x_dense_g_tp", a, b),
        ("tp", b, b),
    ]:
        accum = torch.zeros_like(w1, dtype=torch.float32)
        for key in keys:
            x = xsource[key][target]["input"].cuda()
            g = gsource[key][target]["output_grad"].cuda()
            weight = w1.detach().clone().requires_grad_()
            y = einsum(
                torch.tanh(einsum(x, weight, "t b d, d r -> t b r")),
                w2,
                "t b r, r d -> t b d",
            )
            y.backward(g)
            accum += weight.grad.float()
        replay[label] = accum.cpu()
    report["vjp_swap"] = {
        key: metric(value, replay["dense"]) for key, value in replay.items()
    }
    report["vjp_reference_matches_actual"] = metric(
        replay["dense"], ref["blocks.1.att.w1"]
    )
    # Reproduce the very first divergence using identical dense inputs/weights.
    row = "layer0.self_attention.linear_proj"
    weight = weights["decoder.layers.0.self_attention.linear_proj.weight"].cuda()
    synthetic, observed, dense_outputs = [], [], []
    for key in keys:
        x = a[key][row]["input"].cuda()
        partial = [
            x[..., i * 128 : (i + 1) * 128]
            .contiguous()
            .matmul(weight[:, i * 128 : (i + 1) * 128].contiguous().t())
            for i in range(2)
        ]
        synthetic.append((partial[0] + partial[1]).cpu())
        observed.append(b[key][row]["output"])
        dense_outputs.append(a[key][row]["output"])
    report["first_row_replay_matches_tp"] = metric(
        torch.stack(synthetic), torch.stack(observed)
    )
    report["first_row_split_vs_dense"] = metric(
        torch.stack(synthetic), torch.stack(dense_outputs)
    )

    # Conversion maps this native tensor with its original [D,R] shape.
    (OUT / "precision-diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "layer_metrics"},
            indent=2,
        ),
        flush=True,
    )


def replay():
    before = records("combined", step=3, trace=True)
    after = records("combined-resume", step=3, trace=True)
    if not before or not after:
        return
    report = {}
    for index, (a, b) in enumerate(zip(before, after)):
        differences = []
        assert a["forwards"] == b["forwards"]
        assert [x["key"] for x in a["accum"]] == [x["key"] for x in b["accum"]]
        for micro, (ma, mb) in enumerate(zip(a["accum"], b["accum"])):
            key = ma["key"]
            for label, ga, gb in [
                ("local", a["param_grads"][key], b["param_grads"][key]),
                ("accum", ma["grads"], mb["grads"]),
            ]:
                for name, value in ga.items():
                    if not torch.equal(value, gb[name]):
                        differences.append(
                            dict(
                                micro=micro,
                                phase=label,
                                param=name,
                                metric=metric(gb[name], value),
                                dtype=str(value.dtype),
                            )
                        )
        for key, mods in a["modules"].items():
            for name, entry in mods.items():
                for field, value in entry.items():
                    other = b["modules"][key][name][field]
                    if not torch.equal(value, other):
                        differences.append(
                            dict(
                                micro_key=key,
                                module=name,
                                field=field,
                                metric=metric(other, value),
                            )
                        )
        report[str(index)] = dict(
            dp=a["dp"], tp=a["tp"], pp=a["pp"], differences=differences
        )
    (OUT / "replay-diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "REPLAY DIFF COUNTS",
        {k: len(v["differences"]) for k, v in report.items()},
        flush=True,
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    precision()
    replay()
