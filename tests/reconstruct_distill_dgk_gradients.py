"""Propagate fixed scan alternatives through decay and time-mix VJPs."""

import importlib
import itertools
import json
from pathlib import Path

import torch
from einops import einsum, rearrange

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/distill-numerics-20261023"


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    mix = importlib.import_module("fla.ops.rwkv7.fused_addcmul")
    report = {}
    for sp in (False, True):
        case = "combined-" + ("sp-" if sp else "") + "original-dgktrace-20261026"
        tag = "dp2-tp2-" + ("sp-" if sp else "") + "pp2-bf16"
        traces = {
            r: torch.load(OUT / case / f"trace-step3-rank{r}.pt", weights_only=False)
            for r in range(4, 8)
        }
        snaps = {
            r: torch.load(OUT / case / f"step3-rank{r}.pt", weights_only=False)
            for r in range(4, 8)
        }
        history = torch.load(
            ROOT / "training-smoke/dp2-distill-20261023" / tag / "step3-rank4.pt",
            weights_only=False,
        )
        scans = [
            torch.load(OUT / case / f"dgk-{label}.pt", weights_only=False)
            for label in ("bk32-warps4", "bk64-warps4")
        ]
        raw_mapping_errors = []
        gdecay = {}
        for rank, trace in traces.items():
            for index, entry in enumerate(trace["dgk_kernels"]):
                layer = 1 - index % 2
                key = entry["micro"]
                module = f"layer{layer+2}.self_attention."
                recorded = trace["modules"][key][module + "core_attention"]
                for choice in (0, 1):
                    grad = rearrange(
                        -scans[choice][rank, index].bfloat16(), "b t h n -> t b (h n)"
                    )
                    if choice == 0 and not torch.equal(grad, recorded["operand3_grad"]):
                        raw_mapping_errors.append((rank, index, "wkv"))
                    decay = (
                        trace["modules"][key][module + "bottleneck_decay"]["output"]
                        .cuda()
                        .requires_grad_()
                    )
                    weight = torch.exp(torch.tensor(-0.5)) * decay.sigmoid()
                    g = torch.autograd.grad(weight, decay, grad.cuda())[0].cpu()
                    if choice == 0 and not torch.equal(
                        g,
                        trace["modules"][key][module + "bottleneck_decay"][
                            "output_grad"
                        ],
                    ):
                        raw_mapping_errors.append((rank, index, "sigmoid"))
                    gdecay[rank, key, layer, choice] = g
        assert not raw_mapping_errors, raw_mapping_errors
        cache, local_errors = {}, []
        for rank in (range(4, 8) if sp else (4, 6)):
            peer = rank + (1 if rank % 2 == 0 else -1)
            first, second = sorted((rank, peer))
            trace = traces[rank]
            for choices in itertools.product((0, 1), repeat=2):
                for layer in (0, 1):
                    base = f"decoder.layers.{layer}.self_attention."
                    module = f"layer{layer+2}.self_attention."
                    sums = {
                        suffix: torch.zeros_like(
                            snaps[rank]["grads"][base + suffix], dtype=torch.float32
                        )
                        for suffix in (
                            "bottleneck_decay.weight_down",
                            "bottleneck_decay.weight_up",
                        )
                    }
                    sums["mix_coeff_input"] = torch.zeros_like(
                        snaps[rank]["grads"][base + "mix_coeff_input"],
                        dtype=torch.float32,
                    )
                    for micro in trace["accum"]:
                        key = micro["key"]
                        position = trace["forwards"].index(key)
                        key0, key1 = [
                            traces[r]["forwards"][position] for r in (first, second)
                        ]
                        g = torch.cat(
                            [
                                gdecay[first, key0, layer, choices[0]],
                                gdecay[second, key1, layer, choices[1]],
                            ],
                            -1,
                        )
                        if sp:
                            g = g.chunk(2, 0)[rank % 2]
                        x = (
                            trace["modules"][key][module + "bottleneck_decay"]["input"]
                            .cuda()
                            .requires_grad_()
                        )
                        down = (
                            snaps[rank]["before"][base + "bottleneck_decay.weight_down"]
                            .cuda()
                            .requires_grad_()
                        )
                        up = (
                            snaps[rank]["before"][base + "bottleneck_decay.weight_up"]
                            .cuda()
                            .requires_grad_()
                        )
                        neck = einsum(x, down, "t b d, d r -> t b r").tanh()
                        y = einsum(neck, up, "t b r, r d -> t b d")
                        gx, gd, gu = torch.autograd.grad(y, (x, down, up), g.cuda())
                        hidden = trace["modules"][key][
                            f"layer{layer+2}.input_layernorm"
                        ]["output"].cuda()
                        previous = torch.zeros_like(hidden[:1])
                        if sp and rank % 2:
                            previous = traces[peer]["modules"][key0][
                                f"layer{layer+2}.input_layernorm"
                            ]["output"][-1:].cuda()
                        delta = torch.cat((previous, hidden[:-1]), 0) - hidden
                        zero = torch.zeros_like(gx).reshape(1, -1, gx.shape[-1])
                        coeff = (
                            mix.addcmul_bwd2(
                                zero,
                                gx.reshape_as(zero),
                                zero,
                                zero,
                                zero,
                                zero,
                                delta.reshape_as(zero),
                                True,
                            )[1]
                            .reshape(-1)
                            .bfloat16()
                            .cpu()
                        )
                        values = {
                            "bottleneck_decay.weight_down": gd.cpu(),
                            "bottleneck_decay.weight_up": gu.cpu(),
                        }
                        for suffix, value in values.items():
                            if choices == (0, 0) and not torch.equal(
                                value, trace["param_grads"][key][base + suffix]
                            ):
                                local_errors.append((rank, layer, key, suffix))
                            sums[suffix].add_(value.float())
                        if choices == (0, 0) and not torch.equal(
                            coeff,
                            trace["param_grads"][key][base + "mix_coeff_input"][1],
                        ):
                            local_errors.append((rank, layer, key, "mix"))
                        sums["mix_coeff_input"][1].add_(coeff.float())
                    cache[rank, choices, layer] = sums
        assert not local_errors, local_errors[:10]
        controls = []
        for pattern in itertools.product((0, 1), repeat=4):
            metrics = {}
            for layer in (0, 1):
                base = f"decoder.layers.{layer}.self_attention."
                for suffix in (
                    "bottleneck_decay.weight_down",
                    "bottleneck_decay.weight_up",
                    "mix_coeff_input",
                ):
                    p0, p1 = pattern[:2], pattern[2:]
                    grad = (
                        cache[4, p0, layer][suffix] * 0.5
                        + cache[6, p1, layer][suffix] * 0.5
                    )
                    if sp:
                        grad += (
                            cache[5, p0, layer][suffix] * 0.5
                            + cache[7, p1, layer][suffix] * 0.5
                        )
                    old, now = (
                        history["grads"][base + suffix],
                        snaps[4]["grads"][base + suffix],
                    )
                    if suffix == "mix_coeff_input":
                        grad, old, now = grad[1], old[1], now[1]
                    metrics[base + suffix] = dict(
                        historical_unequal=int((grad != old).sum()),
                        current_unequal=int((grad != now).sum()),
                        historical_max_abs=float((grad - old).abs().max()),
                    )
            unequal = sum(m["historical_unequal"] for m in metrics.values())
            controls.append(
                dict(
                    rank4_to_7_pattern=pattern,
                    historical_unequal=unequal,
                    parameters=metrics,
                )
            )
            print(case, pattern, "historical unequal", unequal, flush=True)
        report[case] = dict(
            default_local_vjp_exact=True,
            scan_to_wkv_gradient_exact=True,
            controls=controls,
        )
    (OUT / "dgk-gradient-reconstruction-20261026.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
