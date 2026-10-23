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


def broadcast_teacher(request, group, src, device):
    """Run request only on src; distribute either top-k or a bounded error.

    Only the final pipeline stage calls this function. On failure all its TP
    ranks raise, so torchrun can terminate pipeline peers waiting in P2P.
    A PP-wide collective inside 1F1B would deadlock across microbatches.
    """
    from rwkv_megatron.online_teacher import TeacherTopK

    source = dist.get_rank() == src
    teacher = None
    error = b''
    if source:
        try:
            teacher = request()
            if not isinstance(teacher, TeacherTopK):
                raise TypeError('teacher response must be TeacherTopK')
            if teacher.token_ids.ndim != 3 or teacher.token_ids.shape != teacher.logprobs.shape:
                raise ValueError('teacher response must have matching [B,S,K] tensors')
            ids = teacher.token_ids.to(device=device, dtype=torch.long).contiguous()
            logprobs = teacher.logprobs.to(device=device, dtype=torch.float32).contiguous()
            header = torch.tensor([1, *ids.shape], dtype=torch.long, device=device)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'.encode('utf-8')[:8192]
            header = torch.tensor([-len(error), 0, 0, 0], dtype=torch.long, device=device)
    else:
        header = torch.empty(4, dtype=torch.long, device=device)
    dist.broadcast(header, src=src, group=group)
    status, *shape = header.tolist()
    if status < 0:
        message = (torch.tensor(list(error), dtype=torch.uint8, device=device) if source else
                   torch.empty(-status, dtype=torch.uint8, device=device))
        dist.broadcast(message, src=src, group=group)
        raise RuntimeError('teacher request failed on TP source: ' + bytes(message.cpu().tolist()).decode('utf-8', errors='replace'))
    if not source:
        ids = torch.empty(shape, dtype=torch.long, device=device)
        logprobs = torch.empty(shape, dtype=torch.float32, device=device)
    dist.broadcast(ids, src=src, group=group)
    dist.broadcast(logprobs, src=src, group=group)
    return TeacherTopK(ids, logprobs)
