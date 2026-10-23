"""Vocabulary-sharded distillation without gathering the student vocabulary.

Every TP rank receives identical targets and returns the same scalar loss.
The backward returns only the gradient of its local vocabulary shard, as in
Megatron's vocabulary-parallel cross entropy (no TP factor in the gradient).
"""

import math

import torch
import torch.distributed as dist


class _TargetLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, valid_vocab_size, group):
        width = logits.shape[-1]
        start = dist.get_rank(group) * width
        local_ids = torch.arange(width, device=logits.device) + start
        scores = logits.float().masked_fill(local_ids >= valid_vocab_size, -torch.inf)
        maximum = scores.max(-1, keepdim=True).values
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        scores = scores - maximum
        probabilities = scores.exp()
        denominator = probabilities.sum(-1, keepdim=True)
        dist.all_reduce(denominator, group=group)
        probabilities /= denominator
        owned = (targets >= start) & (targets < start + width)
        indices = (targets - start).clamp(0, width - 1)
        selected = scores.gather(-1, indices).masked_fill(~owned, 0.0)
        dist.all_reduce(selected, group=group)
        ctx.save_for_backward(probabilities, indices, owned)
        return selected - denominator.log()

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, indices, owned = ctx.saved_tensors
        grad = -probabilities * grad_output.sum(-1, keepdim=True)
        grad.scatter_add_(-1, indices, grad_output.masked_fill(~owned, 0.0))
        return grad, None, None, None


def _validate(logits, valid_vocab_size, group):
    total_vocab = logits.shape[-1] * dist.get_world_size(group)
    if not isinstance(valid_vocab_size, int) or not 1 <= valid_vocab_size <= total_vocab:
        raise ValueError(f"valid_vocab_size must be in [1, {total_vocab}]")


def vocab_parallel_kl(student_logits, teacher, attention_mask, temperature, valid_vocab_size, group):
    """Match online_teacher.kl_divergence_topk on a contiguous TP vocab shard."""
    logits, mask = student_logits, attention_mask
    _validate(logits, valid_vocab_size, group)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = teacher.token_ids.to(device=logits.device, dtype=torch.long)
    logp = teacher.logprobs.to(device=logits.device, dtype=torch.float32) / temperature
    if ids.shape != logp.shape or ids.shape[:2] != logits.shape[:2] or mask.shape != logits.shape[:2]:
        raise ValueError("teacher, student and mask shapes do not align")
    if ((ids < -1) | (ids >= valid_vocab_size)).any():
        raise ValueError("Teacher token IDs exceed student vocabulary; use the same tokenizer")
    usable = (ids >= 0) & torch.isfinite(logp)
    logp = logp.masked_fill(~usable, -torch.inf)
    log_z = torch.logsumexp(logp, -1, keepdim=True)
    rows = mask.to(device=logits.device, dtype=torch.bool) & torch.isfinite(log_z.squeeze(-1))
    if not rows.any():
        return logits.float().sum() * 0.0
    teacher_logp = logp[rows] - log_z[rows]
    probs = teacher_logp.exp().masked_fill(~usable[rows], 0.0)
    teacher_logp = teacher_logp.masked_fill(~usable[rows], 0.0)
    student_logp = _TargetLogProb.apply(
        logits[rows].float() / temperature, ids[rows].clamp_min(0), valid_vocab_size, group
    ).masked_fill(~usable[rows], 0.0)
    return (probs * (teacher_logp - student_logp)).sum(-1).mean() * temperature**2


def vocab_parallel_ce(logits, labels, mask, valid_vocab_size, group):
    """Token-mean CE excluding padded vocabulary rows and masked labels."""
    _validate(logits, valid_vocab_size, group)
    rows = mask.to(device=logits.device, dtype=torch.bool)
    labels = labels.to(device=logits.device, dtype=torch.long)
    targets = labels[rows]
    if ((targets < 0) | (targets >= valid_vocab_size)).any():
        raise ValueError("student CE labels contain IDs outside the effective vocabulary")
    if not rows.any():
        return logits.float().sum() * 0.0
    return -_TargetLogProb.apply(logits[rows].float(), targets[:, None], valid_vocab_size, group).mean()
