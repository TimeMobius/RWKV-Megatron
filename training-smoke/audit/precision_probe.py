"""Audit-only FP32 intermediate arithmetic, keeping matrix multiplications BF16."""
import torch
import torch.nn.functional as F

from v0_fp32 import modified_forward


def install(model,mode):
    assert model.config.tensor_model_parallel_size==1 and model.config.pipeline_model_parallel_size==1
    assert model.config.bf16 and not model.config.sequence_parallel
    if mode == 'key-fused':
        from key_precision import install as install_key
        install_key(model)
        return
    for layer in model.decoder.layers:
        att=layer.self_attention
        replacements=[
            ('key = self.linear_key(mixed_key.contiguous())[0]',
             'key = self.linear_key(mixed_key.contiguous())[0].float()'),
            ('rem_key = key * self.beta','rem_key = key * self.beta.float()'),
            ('rep_key = update_key(key, learn_rate, self.alpha)',
             'rep_key = update_key(key, learn_rate.float(), self.alpha.float()).to(hidden_states.dtype)'),
            ('a = rem_key_norm.neg()','a = rem_key_norm.neg().to(hidden_states.dtype)'),
            ('b = learn_rate * rem_key_norm','b = (learn_rate.float() * rem_key_norm).to(hidden_states.dtype)'),
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
