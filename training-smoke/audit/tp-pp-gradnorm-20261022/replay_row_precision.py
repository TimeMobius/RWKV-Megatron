"""Separate row-GEMM output encoding from collective summation on fixed inputs."""
import json
from pathlib import Path

import torch

from compare_fine import HERE, load, metric

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(8)


def replay(layer, name):
    weight_sources = [torch.load(HERE / "tp2pp4" / f"weights26-rank{r}.pt", mmap=True, weights_only=True)
                      for r in range(2)]
    weights = torch.cat([source["parameters"][f"decoder.layers.{layer}.{name}.weight"]
                         for source in weight_sources], dim=1).cuda()
    reference_input = load("fine-tp4pp2", layer, name, 4, "inputs").cuda()
    actual_input = load("fine-tp2pp2", layer, name, 2, "inputs").cuda()
    fixed = torch.equal(reference_input, actual_input)
    x = reference_input.flatten(0, 1)
    # Independent FP32 operands/accumulation; TF32 is disabled.
    reference = torch.mm(x.float(), weights.float().t())
    rows = {}
    replay_inputs = {}
    for tp, folder in [(2, "fine-tp2pp2"), (4, "fine-tp4pp2")]:
        captured = [torch.load(HERE / folder / f"fine-layer{layer}-{name}-collective-rank{r}.pt",
                               mmap=True, weights_only=True) for r in range(tp)]
        partials = [c["input"].cuda().flatten(0, 1) for c in captured]
        actual = torch.cat([c["output"] for c in captured], dim=0).cuda().flatten(0, 1)
        exact_partials = []
        narrow_parts = []
        reproduction = []
        width = x.shape[-1] // tp
        for rank in range(tp):
            xs = x[:, rank * width:(rank + 1) * width].contiguous()
            ws = weights[:, rank * width:(rank + 1) * width].contiguous()
            high = torch.mm(xs.float(), ws.float().t())
            narrow = torch.mm(xs, ws.t())
            exact_partials.append(high)
            narrow_parts.append(narrow)
            reproduction.append(metric(narrow, partials[rank]))
        high_sum = sum(exact_partials)
        encoded_sum = sum(part.float() for part in partials)
        row = dict(input_matches_other_tp=fixed,
                   production_partial_replay=reproduction,
                   fp32_split_sum_vs_full_fp32=metric(high_sum, reference),
                   partial_output_encoding_vs_full_fp32=metric(encoded_sum, reference),
                   communication_output_vs_fp32_sum_of_captured_partials=metric(actual, encoded_sum),
                   collective_vs_single_final_bf16_rounding=metric(actual, encoded_sum.bfloat16()),
                   final_output_vs_full_fp32=metric(actual, reference),
                   captured_partial_vs_fp32_then_bf16=[metric(p, h.bfloat16()) for p, h in zip(partials, exact_partials)],
                   local_partial_bf16_vs_fp32=[metric(p, h) for p, h in zip(partials, exact_partials)])
        # Candidate production arithmetic with only final BF16 conversion.
        replay_inputs[tp] = dict(high=high_sum.bfloat16().cpu(), encoded=encoded_sum.bfloat16().cpu(), actual=actual.cpu())
        rows[str(tp)] = row
        del captured, partials, actual, exact_partials, narrow_parts, high_sum, encoded_sum
    rows["between_tp"] = {name: metric(replay_inputs[2][name], replay_inputs[4][name])
                          for name in ["high", "encoded", "actual"]}
    torch.save(dict(input=x.cpu(), weight=weights.cpu(), candidates=replay_inputs),
               HERE / f"fixed-row-layer{layer}-{name}.pt")
    del reference_input, actual_input, weights, reference
    torch.cuda.empty_cache()
    return rows


result = {}
for layer, name in [(0, "self_attention.linear_proj")]:
    result[f"layer{layer}-{name}"] = replay(layer, name)
    print(json.dumps(result, indent=2), flush=True)
(HERE / "row-precision-replay.json").write_text(json.dumps(result, indent=2) + "\n")
