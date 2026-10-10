"""Audit-only key fusion: BF16 inputs/outputs, FP32 intermediate arithmetic."""
import torch
import torch.nn.functional as F


def key_fp32(key, learn, beta, alpha, head_size=64):
    k, a = key.float(), learn.float()
    kk = F.normalize((k * beta.float()).reshape(*key.shape[:-1], -1, head_size), dim=-1).reshape_as(k)
    rep = k * (1 + (a - 1) * alpha.float())
    return rep.to(key.dtype), (-kk).to(key.dtype), (a * kk).to(key.dtype)


# Full graph capture fails visibly instead of silently falling back to eager.
# Compiler contraction/reduction choices are checked against the independent
# formula; bit equality with eager arithmetic is not assumed.
key_compiled = torch.compile(key_fp32, fullgraph=True, dynamic=False)


def install(model):
    from v0_fp32 import modified_forward
    assert model.config.tensor_model_parallel_size == 1
    assert model.config.pipeline_model_parallel_size == 1
    assert model.config.bf16 and not model.config.sequence_parallel
    for layer in model.decoder.layers:
        att = layer.self_attention
        att._audit_key_compiled = key_compiled
        modified_forward(att, [
            ('rep_key, a, b = prepare_key(key, learn_rate, self.beta, self.alpha, self.head_size, fusion=getattr(self.config, "rwkv_key_fusion", None))',
             'rep_key, a, b = self._audit_key_compiled(key, learn_rate, self.beta, self.alpha, self.head_size)'),
        ])
