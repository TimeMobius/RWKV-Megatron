"""Replay historical decay VJPs with fixed operands and GEMM precision controls."""

import json
import sys
from pathlib import Path

import torch
from einops import einsum

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-numerics-20261023"


def load(case, rank, trace=False):
    return torch.load(
        OUT / case / f'{"trace-" if trace else ""}step3-rank{rank}.pt',
        weights_only=False,
    )


def vjp(x, down, up, g, mode):
    down, up = (
        down.detach().clone().requires_grad_(),
        up.detach().clone().requires_grad_(),
    )
    if mode == "mm":
        neck = (x.flatten(0, 1) @ down).tanh()
        output = (neck @ up).reshape(g.shape)
    else:
        neck = einsum(x, down, "t b d, d r -> t b r").tanh()
        output = einsum(neck, up, "t b r, r d -> t b d")
    gd, gu = torch.autograd.grad(output, (down, up), g)
    if mode in ("wgrad-fp32", "wgrad-fp64"):
        dtype = torch.float64 if mode == "wgrad-fp64" else torch.float32
        # Keep the forward and propagated BF16 derivative fixed, changing only
        # each weight-gradient dot product accumulator/output control.
        gn = g.flatten(0, 1) @ up.detach().t()
        gz = torch.ops.aten.tanh_backward(gn.reshape(neck.shape), neck.detach())
        gd = (x.flatten(0, 1).to(dtype).t() @ gz.flatten(0, 1).to(dtype)).to(down.dtype)
        gu = (neck.flatten(0, 1).to(dtype).t() @ g.flatten(0, 1).to(dtype)).to(up.dtype)
    return gd, gu


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {}
    for sp in (False, True):
        case = "combined-sp-original" if sp else "combined-original"
        tag = "dp2-tp2-sp-pp2-bf16" if sp else "dp2-tp2-pp2-bf16"
        traces = {rank: load(case, rank, trace=True) for rank in range(4, 8)}
        snaps = {rank: load(case, rank) for rank in range(4, 8)}
        historical = torch.load(
            ROOT / "training-smoke/dp2-distill-20261023" / tag / "step3-rank4.pt",
            weights_only=False,
        )
        case_report = {}
        for mode in ("default", "no-reduced", "mm", "wgrad-fp32", "wgrad-fp64"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                mode != "no-reduced"
            )
            local = {}
            mismatches = []
            for rank in (4, 6) if not sp else range(4, 8):
                tp, peer = rank % 2, rank + (1 if rank % 2 == 0 else -1)
                t = traces[rank]
                for layer in (0, 1):
                    param = f"decoder.layers.{layer}.self_attention.bottleneck_decay"
                    module = f"layer{layer+2}.self_attention.bottleneck_decay"
                    down = snaps[rank]["before"][param + ".weight_down"].cuda()
                    up = snaps[rank]["before"][param + ".weight_up"].cuda()
                    values = []
                    for micro in t["accum"]:
                        key = micro["key"]
                        entry = t["modules"][key][module]
                        peer_key = traces[peer]["forwards"][t["forwards"].index(key)]
                        other = traces[peer]["modules"][peer_key][module]
                        chunks = [entry["output_grad"], other["output_grad"]]
                        if tp:
                            chunks.reverse()
                        g = torch.cat(chunks, -1)
                        if sp:
                            g = g.chunk(2, 0)[tp]
                        result = vjp(entry["input"].cuda(), down, up, g.cuda(), mode)
                        values.append(tuple(v.cpu() for v in result))
                        for suffix, value in zip(("weight_down", "weight_up"), result):
                            target = t["param_grads"][key][param + "." + suffix]
                            n = int((value.cpu() != target).sum())
                            if n:
                                mismatches.append(
                                    dict(
                                        rank=rank,
                                        micro=key,
                                        parameter=param + "." + suffix,
                                        unequal=n,
                                    )
                                )
                    totals = [
                        torch.zeros_like(values[0][index], dtype=torch.float32)
                        for index in (0, 1)
                    ]
                    for value in values:
                        for index in (0, 1):
                            totals[index].add_(value[index].float())
                    local[rank, layer] = tuple(totals)
            gradients = {}
            for layer in (0, 1):
                for index, suffix in enumerate(("weight_down", "weight_up")):
                    param = f"decoder.layers.{layer}.self_attention.bottleneck_decay.{suffix}"
                    if sp:
                        # Production first averages DP, then sums SP partials.
                        grad = (
                            local[4, layer][index] * 0.5 + local[6, layer][index] * 0.5
                        ) + (
                            local[5, layer][index] * 0.5 + local[7, layer][index] * 0.5
                        )
                    else:
                        grad = (
                            local[4, layer][index] * 0.5 + local[6, layer][index] * 0.5
                        )
                    old, current = historical["grads"][param], snaps[4]["grads"][param]
                    gradients[param] = dict(
                        current_unequal=int((grad != current).sum()),
                        historical_unequal=int((grad != old).sum()),
                        historical_max_abs=float((grad - old).abs().max()),
                    )
            case_report[mode] = dict(local_mismatches=mismatches, aggregate=gradients)
            print(
                case,
                mode,
                "local mismatches",
                len(mismatches),
                "aggregate",
                gradients,
                flush=True,
            )
        report[case] = case_report
    (OUT / "decay-vjp-20261026.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
