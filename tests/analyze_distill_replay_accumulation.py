"""Replay captured BF16 microgradients through FP32 accumulation and DP averaging."""

import itertools
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-numerics-20261023"
OLD = ROOT / "training-smoke/dp2-distill-20261023/dp2-tp2-pp2-bf16"


def warm_control():
    report = {}
    for case in ("combined-original", "combined-sp-original"):
        differences = []

        def visit(a, b, path):
            if isinstance(a, torch.Tensor):
                assert a.shape == b.shape and a.dtype == b.dtype
                if not torch.equal(a, b):
                    differences.append(
                        dict(
                            path=path,
                            unequal=int((a != b).sum()),
                            max_abs=float((a.float() - b.float()).abs().max()),
                        )
                    )
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    visit(a[key], b[key], f"{path}/{key}")
            elif isinstance(a, (tuple, list)):
                assert len(a) == len(b)
                for index, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f"{path}/{index}")
            else:
                assert a == b, path

        for rank in range(8):
            a = torch.load(
                OUT / case / f"trace-step3-rank{rank}.pt", weights_only=False
            )
            b = torch.load(
                OUT / (case + "-rewarm") / f"trace-step3-rank{rank}.pt",
                weights_only=False,
            )
            for field in ("forwards", "modules", "param_grads", "accum"):
                visit(a[field], b[field], f"rank{rank}/{field}")
        report[case] = dict(differences=differences, bit_identical=not differences)
    (OUT / "warm-runtime-control.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "DIRECT VS REWARM",
        {case: len(info["differences"]) for case, info in report.items()},
    )


def main():
    traces = [
        torch.load(
            OUT / "combined-original" / f"trace-step3-rank{rank}.pt", weights_only=False
        )
        for rank in (4, 6)
    ]
    actual = torch.load(
        OUT / "combined-original" / "step3-rank4.pt", weights_only=False
    )
    historical = torch.load(OLD / "step3-rank4.pt", weights_only=False)
    report = {}
    suspects = [
        name
        for name, value in actual["grads"].items()
        if not torch.equal(value, historical["grads"][name])
    ]
    for name in suspects:
        sequences, variants, accum_exact = [], [], []
        for trace in traces:
            sequence = [
                trace["param_grads"][micro["key"]][name] for micro in trace["accum"]
            ]
            assert all(value.dtype == torch.bfloat16 for value in sequence)
            total = torch.zeros_like(sequence[0], dtype=torch.float32)
            for micro, value in zip(trace["accum"], sequence):
                total.add_(value.float())
                accum_exact.append(torch.equal(total, micro["grads"][name]))
            sequences.append(sequence)
            candidate = []
            for order in itertools.permutations(range(len(sequence))):
                total = torch.zeros_like(sequence[0], dtype=torch.float32)
                for index in order:
                    total.add_(sequence[index].float())
                candidate.append((order, total))
            variants.append(candidate)
        # Megatron scales each local accumulation by 1/DP before NCCL SUM.
        current = (variants[0][0][1] * 0.5) + (variants[1][0][1] * 0.5)
        target = historical["grads"][name]
        best = None
        for (order0, value0), (order1, value1) in itertools.product(*variants):
            proposed = value0 * 0.5 + value1 * 0.5
            count = int((proposed != target).sum())
            if best is None or count < best["unequal_elements"]:
                best = dict(
                    unequal_elements=count,
                    dp0_micro_order=order0,
                    dp1_micro_order=order1,
                )
        all_terms = [value.double() for sequence in sequences for value in sequence]
        ideal = (torch.stack(all_terms).sum(0) * 0.5).float()
        report[name] = dict(
            per_micro_accumulation_exact=all(accum_exact),
            captured_final_reduction_replay_exact=torch.equal(
                current, actual["grads"][name]
            ),
            current_vs_historical_unequal=int((current != target).sum()),
            best_reordered_accumulation=best,
            ideal_fp64_sum_vs_current_unequal=int((ideal != current).sum()),
            ideal_fp64_sum_vs_historical_unequal=int((ideal != target).sum()),
        )
    (OUT / "accumulation-diagnosis.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
    warm_control()
