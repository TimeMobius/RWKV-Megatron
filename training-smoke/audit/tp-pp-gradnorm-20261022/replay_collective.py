"""Reproduce captured layer0 row-projection communication in isolation."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
torch.set_num_threads(8)
dist.init_process_group("nccl")
tp = dist.get_world_size()
folder = HERE / f"fine-tp{tp}pp2"
data = torch.load(folder / f"fine-layer0-self_attention.linear_proj-collective-rank{rank}.pt",
                  mmap=True, weights_only=True)
partial = data["input"].cuda()
reference = data["output"].cuda()
bf16_result = torch.empty_like(reference)
dist.reduce_scatter_tensor(bf16_result, partial)
fp32_result = torch.empty_like(reference, dtype=torch.float32)
dist.reduce_scatter_tensor(fp32_result, partial.float())
all_partials = [torch.load(folder / f"fine-layer0-self_attention.linear_proj-collective-rank{i}.pt",
                           mmap=True, weights_only=True)["input"] for i in range(tp)]
single_round = sum(x.float() for x in all_partials).bfloat16().chunk(tp, dim=0)[rank].cuda()
ideal_sum = sum(x.double() for x in all_partials).chunk(tp, dim=0)[rank].cuda()
rows = dict(rank=rank, tp=tp,
            bf16_collective_matches_training_capture=torch.equal(bf16_result, reference),
            fp32_collective_matches_single_final_rounding=torch.equal(fp32_result.bfloat16(), single_round),
            fp32_collective_vs_single_round_unequal_elements=int((fp32_result.bfloat16() != single_round).sum()),
            bf16_vs_exact_sum_relative_l2=float((bf16_result.double()-ideal_sum).norm()/ideal_sum.norm()),
            fp32_vs_exact_sum_relative_l2=float((fp32_result.double()-ideal_sum).norm()/ideal_sum.norm()),
            fp32_then_bf16_vs_ideal_bf16_unequal_elements=int((fp32_result.bfloat16() != ideal_sum.bfloat16()).sum()),
            bf16_vs_fp32_collective_relative_l2=float((bf16_result.float()-fp32_result).norm()/fp32_result.norm()),
            bf16_vs_fp32_then_bf16_unequal_elements=int((bf16_result != fp32_result.bfloat16()).sum()))
gathered = [None for _ in range(tp)]
dist.all_gather_object(gathered, rows)
if rank == 0:
    (HERE / f"collective-replay-tp{tp}.json").write_text(json.dumps(gathered, indent=2)+"\n")
    print(json.dumps(gathered, indent=2), flush=True)
dist.destroy_process_group()
