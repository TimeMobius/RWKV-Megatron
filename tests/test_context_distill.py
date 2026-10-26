"""CP boundary targets and independently normalized masked objectives."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rwkv_megatron.context_distill import finalize_context_loss, shard_teacher
from rwkv_megatron.online_teacher import TeacherTopK, kl_divergence_topk


def worker(rank, size, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=size)
    try:
        generator = torch.Generator().manual_seed(1984)
        logits = torch.randn(2, 16, 11, generator=generator)
        labels = torch.randint(0, 11, (2, 16), generator=generator)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[0, 10:] = False
        mask[1, 5:] = False
        ids = torch.randint(0, 11, (2, 16, 3), generator=generator)
        lp = torch.randn(2, 16, 3, generator=generator)
        ids[:, 3:5] = -1
        teacher = TeacherTopK(ids, lp)
        start, end = rank * (16 // size), (rank + 1) * (16 // size)
        target = shard_teacher(teacher, rank, size)
        assert torch.equal(
            target.token_ids[:, 1:], ids[:, start + 1 : min(end + 1, 16)]
        )
        for ce_weight in (0.0, 0.3):
            full = logits.clone().requires_grad_()
            reference_kl = kl_divergence_topk(
                full[:, :-1],
                TeacherTopK(ids[:, 1:], lp[:, 1:]),
                mask[:, :-1],
                temperature=2.0,
                valid_vocab_size=11,
            )
            reference_ce = F.cross_entropy(full[mask], labels[mask])
            objective = reference_kl + ce_weight * reference_ce
            objective.backward()
            local = logits[:, start:end].clone().requires_grad_()
            length = target.token_ids.shape[1] - 1
            local_mask = mask[:, start : start + length]
            kl = kl_divergence_topk(
                local[:, :length],
                TeacherTopK(target.token_ids[:, 1:], target.logprobs[:, 1:]),
                local_mask,
                temperature=2.0,
                valid_vocab_size=11,
            )
            ce_mask = mask[:, start:end]
            ce = (
                F.cross_entropy(local[ce_mask], labels[:, start:end][ce_mask])
                if ce_mask.any()
                else local.sum() * 0
            )
            usable = (
                (target.token_ids[:, 1:] >= 0) & torch.isfinite(target.logprobs[:, 1:])
            ).any(-1)
            loss, count, metrics = finalize_context_loss(
                kl,
                ce,
                (local_mask & usable).float().sum(),
                ce_mask.float().sum(),
                1.0,
                ce_weight,
                dist.group.WORLD,
                local_mask.float().sum(),
            )
            assert count.item() == mask[:, :-1].sum().item()
            torch.testing.assert_close(loss / size / count, objective.detach())
            torch.testing.assert_close(
                metrics["distill/kl_loss"][0] / metrics["distill/kl_loss"][1],
                reference_kl.detach(),
            )
            torch.testing.assert_close(
                metrics["distill/student_ce_loss"][0]
                / metrics["distill/student_ce_loss"][1],
                reference_ce.detach(),
            )
            (loss / count / size).backward()
            torch.testing.assert_close(
                local.grad, full.grad[:, start:end], rtol=1e-5, atol=1e-7
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
def test_context_objective(tmp_path, size):
    mp.spawn(worker, args=(size, Path(tmp_path / "rdzv").as_uri()), nprocs=size)
