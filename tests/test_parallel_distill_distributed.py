"""Run with torchrun --standalone --nproc-per-node={2,4} this_file.py."""

import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rwkv_megatron.online_teacher import TeacherTopK, kl_divergence_topk
from rwkv_megatron.parallel_distill import vocab_parallel_ce, vocab_parallel_kl


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    group = dist.group.WORLD
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(20261023)
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for vocab in (19, 7):  # second case includes ranks with entirely padded vocab
            for empty in (False, True):
                for ce_weight in (0.0, 0.3):
                    dense = torch.randn(2, 5, 24, device='cuda', dtype=dtype, requires_grad=True)
                    with torch.no_grad():
                        dense[..., vocab:] = 100  # padding must contribute neither mass nor gradients
                    ids = torch.randint(vocab, (2, 5, 4), device='cuda')
                    ids[..., 1] = ids[..., 0]  # duplicate top-k entries exercise scatter_add
                    ids[0, 1] = -1
                    lp = torch.randn(2, 5, 4, device='cuda')
                    lp[1, 0, 2] = -torch.inf
                    teacher = TeacherTopK(ids, lp)
                    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 1, 0]], device='cuda')
                    if empty:
                        mask.zero_()
                    labels = torch.randint(vocab, (2, 5), device='cuda').masked_fill(~mask.bool(), -100)
                    local = dense.detach().chunk(size, -1)[rank].contiguous().requires_grad_()
                    actual_kl = vocab_parallel_kl(local, teacher, mask, 2.0, vocab, group)
                    expected_kl = kl_divergence_topk(dense, teacher, mask, 2.0, vocab)
                    actual_ce = vocab_parallel_ce(local, labels, mask, vocab, group)
                    expected_ce = (F.cross_entropy(dense[mask.bool()][:, :vocab].float(), labels[mask.bool()])
                                   if mask.any() else dense.float().sum() * 0)
                    actual = actual_kl + ce_weight * actual_ce
                    expected = expected_kl + ce_weight * expected_ce
                    actual.backward()
                    expected.backward()
                    reference_grad = dense.grad.chunk(size, -1)[rank]
                    torch.testing.assert_close(actual_kl, expected_kl, rtol=1e-5, atol=2e-6)
                    torch.testing.assert_close(actual_ce, expected_ce, rtol=1e-5, atol=2e-6)
                    torch.testing.assert_close(local.grad, reference_grad,
                                               rtol=0.008 if dtype == torch.bfloat16 else 1e-5,
                                               atol=0.00025 if dtype == torch.bfloat16 else 2e-6)
                    assert torch.count_nonzero(local.grad[..., max(0, vocab - rank * (24 // size)):]) == 0
                    results.append(dict(dtype=str(dtype), vocab=vocab, empty=empty, ce_weight=ce_weight,
                                        loss_abs=float((actual-expected).abs().detach()),
                                        grad_max_abs=float((local.grad-reference_grad).abs().max())))
    if rank == 0:
        print(json.dumps(dict(tp=size, cases=len(results), results=results)), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
