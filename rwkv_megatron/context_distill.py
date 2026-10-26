"""Contiguous context shards preserve full-context teacher alignment and means."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .online_teacher import TeacherTopK


@dataclass
class ContextTeacherInput:
    """Teacher input before student tokens are split into context shards."""

    tokens: torch.Tensor
    loss_mask: torch.Tensor
    prompts: object


def shard_teacher(teacher, rank, size):
    """Include the next shard's first target for the boundary student logit."""
    length = teacher.token_ids.shape[1]
    if length % size:
        raise ValueError("Teacher sequence must divide evenly across CP ranks")
    width = length // size
    start, stop = rank * width, min((rank + 1) * width + 1, length)
    return TeacherTopK(
        teacher.token_ids[:, start:stop].contiguous(),
        teacher.logprobs[:, start:stop].contiguous(),
    )


def finalize_context_loss(
    kl, ce, kl_count, ce_count, kl_weight, ce_weight, group, token_count=None
):
    """Return Megatron's numerator/count with the derivative of a global mean.

    KL and CE have different masks/counts. Normalize them independently before
    converting back to the common numerator/count consumed by the schedule.
    DDP averages gradients over CP; the CP multiplier restores their sum.
    """
    if token_count is None:
        token_count = kl_count
    counts = torch.stack(
        (kl_count.detach(), ce_count.detach(), token_count.detach())
    ).clone()
    dist.all_reduce(counts, group=group)
    global_kl, global_ce, global_tokens = counts.unbind()
    local = kl_weight * kl * kl_count * global_tokens / global_kl.clamp_min(1)
    local = local + ce_weight * ce * ce_count * global_tokens / global_ce.clamp_min(1)
    reporting = torch.stack(
        (local.detach(), kl.detach() * kl_count, ce.detach() * ce_count)
    )
    dist.all_reduce(reporting, group=group)
    global_value = reporting[0] + (local - local.detach())
    metrics = {
        "lm loss": torch.stack((reporting[0], global_tokens)),
        "distill/kl_loss": torch.stack(
            (reporting[1] * global_tokens / global_kl.clamp_min(1), global_tokens)
        ),
        "distill/student_ce_loss": torch.stack((reporting[2], global_ce)),
    }
    if ce_weight:
        # Preserve the pre-existing CE monitoring key, normalized by its own
        # token count rather than by local shard length.
        metrics["distill/ce_loss"] = torch.stack((reporting[2], global_ce))
    return (
        global_value * dist.get_world_size(group),
        global_tokens.to(torch.int),
        metrics,
    )
