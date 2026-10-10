"""Per-token RWKV max-logit gradient, retaining Megatron's CE and scaling."""
import torch
import torch.distributed as dist


@torch.no_grad()
def max_logit_stats(logits, tp_group=None):
    """Capture before CE: some CE backends overwrite FP32 logits in place."""
    max_value, index = logits.float().max(dim=-1)
    local_vocab = logits.shape[-1]
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        rank = dist.get_rank(tp_group)
        local_max = max_value.clone()
        dist.all_reduce(max_value, op=dist.ReduceOp.MAX, group=tp_group)
        # A tie selects the first GLOBAL vocabulary index, as torch.max does
        # on the unsharded tensor. Exactly one shard owns the extra gradient.
        global_index = torch.where(local_max == max_value, index + rank * local_vocab,
                                   dist.get_world_size(tp_group) * local_vocab)
        dist.all_reduce(global_index, op=dist.ReduceOp.MIN, group=tp_group)
        local_index = global_index - rank * local_vocab
        owned = (local_index >= 0) & (local_index < local_vocab)
        max_value = torch.where(owned, max_value, 0)
        index = local_index.clamp(0, local_vocab - 1)
    return max_value, index


class _L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_loss, logits, coefficient, stats):
        ctx.save_for_backward(*stats)
        ctx.logits_shape = logits.shape
        ctx.logits_dtype = logits.dtype
        ctx.coefficient = coefficient
        return token_loss

    @staticmethod
    def backward(ctx, grad_output):
        max_value, index = ctx.saved_tensors
        # No token-count division here. The per-token upstream gradient already
        # includes the mask, averaging, microbatch scaling and any loss scale.
        extra = max_value * ctx.coefficient * grad_output.transpose(0, 1).float()
        grad_logits = torch.zeros(ctx.logits_shape, device=grad_output.device,
                                  dtype=ctx.logits_dtype)
        grad_logits.scatter_(-1, index.unsqueeze(-1), extra.to(ctx.logits_dtype).unsqueeze(-1))
        return grad_output, grad_logits, None, None


def wrap_token_loss(token_loss, logits, coefficient, tp_group=None, stats=None):
    """Wrap [B,S] CE without changing its forward value.

    tp_group must describe vocabulary shards; pass None for full-vocabulary
    logits, even if the model itself uses TP. Masked tokens receive no penalty
    because masking is applied to token_loss by the existing training entry.
    """
    if token_loss.shape != logits.shape[:2][::-1]:
        raise ValueError('L2Wrap expects loss [batch, sequence] and logits [sequence, batch, vocab]')
    if stats is None:
        stats = max_logit_stats(logits, tp_group)
    return _L2Wrap.apply(token_loss, logits, coefficient, stats)
