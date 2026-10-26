"""Replay captured WKV reverse scans under valid FLA launch configurations."""

import importlib
import json
from pathlib import Path

import torch
import triton
from triton.runtime.autotuner import Autotuner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-numerics-20261023"


def kernel():
    value = importlib.import_module(
        "fla.ops.generalized_delta_rule.dplr.chunk_A_bwd"
    ).chunk_dplr_bwd_dgk_kernel
    while not isinstance(value, Autotuner):
        value = value.fn
    return value.fn


def launch(fn, operands, block, warps, stages):
    values = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in operands.items()
    }
    expected = values["dgk_output"]
    values["dgk_output"] = torch.empty_like(expected)
    values["BK"] = block
    grid = (
        triton.cdiv(values["T"], values["BT"]),
        triton.cdiv(values["K"], block),
        values["dgk"].shape[0] * values["H"],
    )
    fn[grid](**values, num_warps=warps, num_stages=stages)
    return values["dgk_output"].cpu()


def main():
    fn = kernel()
    report = {}
    for sp in (False, True):
        case = "combined-" + ("sp-" if sp else "") + "original-dgktrace-20261026"
        traces = {
            rank: torch.load(
                OUT / case / f"trace-step3-rank{rank}.pt", weights_only=False
            )
            for rank in range(4, 8)
        }
        controls = {}
        for block in (32, 64):
            for warps in (2, 4, 8, 16, 32):
                diffs = []
                results = {}
                for rank, trace in traces.items():
                    for index, entry in enumerate(trace["dgk_kernels"]):
                        expected = entry["operands"]["dgk_output"]
                        result = launch(fn, entry["operands"], block, warps, 2)
                        fp32_diff = int((result != expected).sum())
                        bf16_diff = int(
                            (result.bfloat16() != expected.bfloat16()).sum()
                        )
                        results[rank, index] = result
                        if fp32_diff:
                            diffs.append(
                                dict(
                                    rank=rank,
                                    kernel=index,
                                    micro=entry["micro"],
                                    fp32_unequal=fp32_diff,
                                    bf16_unequal=bf16_diff,
                                    fp32_max_abs=float((result - expected).abs().max()),
                                )
                            )
                label = f"bk{block}-warps{warps}"
                controls[label] = dict(
                    differences=diffs,
                    bf16_unequal=sum(v["bf16_unequal"] for v in diffs),
                    fp32_unequal=sum(v["fp32_unequal"] for v in diffs),
                )
                # Keep the scan outputs local for downstream VJP reconstruction.
                torch.save(results, OUT / case / f"dgk-{label}.pt")
                print(
                    case,
                    label,
                    "fp32",
                    controls[label]["fp32_unequal"],
                    "bf16",
                    controls[label]["bf16_unequal"],
                    flush=True,
                )
        report[case] = controls
    (OUT / "dgk-kernel-controls-20261026.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
