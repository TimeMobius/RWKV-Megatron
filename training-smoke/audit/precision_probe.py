"""Audit-only FP32 intermediate arithmetic, keeping matrix multiplications BF16."""
import torch
import torch.nn.functional as F

from v0_fp32 import modified_forward


def key_eager_high(key, learn, beta, alpha, head_size):
    from rwkv_megatron.rwkv7_layer_specs import update_key
    k = key.float()
    kk = F.normalize((k * beta.float()).reshape(*key.shape[:-1], -1, head_size), dim=-1).reshape_as(k)
    return (update_key(k, learn.float(), alpha.float()).to(key.dtype),
            (-kk).to(key.dtype), (learn.float() * kk).to(key.dtype))


def key_legacy(key, learn, beta, alpha, head_size):
    from rwkv_megatron.rwkv7_layer_specs import update_key
    kk = F.normalize((key * beta).reshape(*key.shape[:-1], -1, head_size), dim=-1).reshape_as(key)
    return update_key(key, learn, alpha), -kk, learn * kk


def install_legacy(model):
    for layer in model.decoder.layers:
        att = layer.self_attention
        att._audit_key_legacy = key_legacy
        modified_forward(att, [
            ('rep_key, a, b = prepare_key(key, learn_rate, self.beta, self.alpha, self.head_size, fusion=getattr(self.config, "rwkv_key_fusion", None))',
             'rep_key, a, b = self._audit_key_legacy(key, learn_rate, self.beta, self.alpha, self.head_size)'),
        ])


def install(model,mode):
    assert model.config.tensor_model_parallel_size==1 and model.config.pipeline_model_parallel_size==1
    assert model.config.bf16 and not model.config.sequence_parallel
    if mode == 'key-fused':
        from key_precision import install as install_key
        install_key(model)
        return
    for layer in model.decoder.layers:
        att=layer.self_attention
        att._audit_key_eager_high = key_eager_high
        replacements=[
            ('rep_key, a, b = prepare_key(key, learn_rate, self.beta, self.alpha, self.head_size, fusion=getattr(self.config, "rwkv_key_fusion", None))',
             'rep_key, a, b = self._audit_key_eager_high(key, learn_rate, self.beta, self.alpha, self.head_size)'),
        ]
        if mode=='all':
            # Original linear outputs stay BF16; bias addition and sigmoid are
            # FP32, following the official fused scalar operators.
            for bottleneck in (att.bottleneck_learn,att.bottleneck_v):
                modified_forward(bottleneck,[
                    ('return tp_expanded + self.bias if self.use_bias else tp_expanded',
                     'return tp_expanded.float() + self.bias.float() if self.use_bias else tp_expanded'),
                ])
            replacements.extend([
                ('learn_rate = torch.sigmoid(self.bottleneck_learn(mixed_learn_rate.contiguous()))',
                 'learn_rate = torch.sigmoid(self.bottleneck_learn(mixed_learn_rate.contiguous())).to(hidden_states.dtype)'),
                ('value = torch.lerp(value, v0_shard, vt)',
                 'value = torch.lerp(value.float(), v0_shard.float(), vt.float()).to(value.dtype)'),
                ('prod = r * k * p','prod = r.float() * k.float() * p.float()'),
                ('ut_intermediate = summed * v','ut_intermediate = summed * v.float()'),
                ('self.linear_proj(output * gate)',
                 'self.linear_proj((output * gate.float()).to(gate.dtype))'),
            ])
            def norm32(x,module=att.group_norm):
                return F.group_norm(x.float(),module.num_groups,module.weight.float(),module.bias.float(),module.eps)
            att.group_norm.forward=norm32
        modified_forward(att,replacements)
