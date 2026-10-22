"""Replay identical 4K WKV inputs/adjoints with 40 versus 2x20 heads."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "3rdparty/flash-linear-attention-260115"))
from fla.ops.rwkv7 import chunk_rwkv7

HERE = Path(__file__).resolve().parent


def core(values):
    r, k, v, w, a, b = [x.transpose(0, 1).reshape(x.shape[1], x.shape[0], -1, 64).contiguous()
                         for x in values]
    result = chunk_rwkv7(r, -w, k, v, a, b, scale=1.0, safe_gate=True, chunk_size=64)[0]
    return result.flatten(2).transpose(0, 1).contiguous()


rows = []
for layer, rank in [(15, 1), (19, 0), (23, 3)]:
    prefix = HERE / "replay-tp2pp4-common-tp2" / f"layer{layer}-wkv-mb0-rank{rank}"
    fixture = torch.load(str(prefix) + ".pt", mmap=True, weights_only=True)
    values = [x.cuda().detach().requires_grad_() for x in fixture["inputs"]]
    adjoint = torch.load(str(prefix) + "-grad.pt", weights_only=True).cuda()
    output = core(values)
    grads = torch.autograd.grad(output, values, adjoint)
    independent = [x.detach().requires_grad_() for x in values]
    partitioned = torch.cat([core([x.chunk(2, dim=-1)[rank].contiguous() for x in independent])
                             for rank in range(2)], dim=-1)
    split_grads = torch.autograd.grad(partitioned, independent, adjoint)
    row = dict(layer=layer, output_bitwise_equal=torch.equal(output, partitioned),
               input_gradients_bitwise_equal=[torch.equal(a, b) for a, b in zip(grads, split_grads)],
               output_max_abs=float((output.float()-partitioned.float()).abs().max()),
               gradient_max_abs=[float((a.float()-b.float()).abs().max()) for a, b in zip(grads, split_grads)])
    rows.append(row)
    print(row, flush=True)
    del values, adjoint, output, grads, independent, partitioned, split_grads
    torch.cuda.empty_cache()
result = dict(passed=all(row["output_bitwise_equal"] and all(row["input_gradients_bitwise_equal"]) for row in rows),
              scope="Identical captured inputs and incoming gradients, seq4096; only WKV head partition changes", rows=rows)
(HERE / "wkv-head-split.json").write_text(json.dumps(result, indent=2)+"\n")
assert result["passed"], result
