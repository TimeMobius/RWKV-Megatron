"""Single-GPU diagnostic: preserve BF16 math, accumulate value carrier in FP32.

No production source edits. The packed carrier is FP32, with hidden restored to
BF16 before each operation. Layer-zero WKV/correction cast separately, allowing
all value paths to meet in FP32 before one cast back to the projection gradient.
"""
import inspect
import textwrap
from types import MethodType

import torch
from megatron.core import parallel_state as ps


def modified_forward(module, replacements):
    original=type(module).forward
    source=getattr(module,'_audit_forward_source',None)
    if source is None:source=textwrap.dedent(inspect.getsource(original))
    for before,after in replacements:
        assert source.count(before)==1, before
        source=source.replace(before,after)
    namespace=dict(original.__globals__)
    exec(compile(source,'<audit-v0-fp32>', 'exec'),namespace)
    module.forward=MethodType(namespace['forward'],module)
    module._audit_forward_source=source


def install(model):
    assert ps.get_tensor_model_parallel_world_size()==1
    assert ps.get_pipeline_model_parallel_world_size()==1
    assert not model.config.sequence_parallel and model.config.bf16
    for layer in model.decoder.layers:
        modified_forward(layer.self_attention,[
            ('hidden_states, v0 = hidden_states[0], hidden_states[1]',
             'hidden_states, v0 = hidden_states[0].to(torch.bfloat16), hidden_states[1]'),
            ('if v0 is None and self.layer_number == 1 and pp_rank == 0:',
             '_audit_source = value.float() if self.layer_number == 1 else None\n'
             '    if _audit_source is not None:\n'
             '        if _audit_source.requires_grad:\n'
             '            _audit_source.retain_grad()\n'
             '        self.audit_v0_source = _audit_source\n\n'
             '    if v0 is None and self.layer_number == 1 and pp_rank == 0:'),
            ('v0 = gather_from_tensor_model_parallel_region(value)',
             'v0 = gather_from_tensor_model_parallel_region(_audit_source)'),
            ('v0_shard = v0[:, :, start_idx:end_idx]',
             'v0_shard = v0[:, :, start_idx:end_idx].to(torch.bfloat16)'),
            ('output = self.core_attention(receptance, rep_key, value, weight, a, b)',
             '_audit_core_value = _audit_source.to(torch.bfloat16) if _audit_source is not None else value\n'
             '    output = self.core_attention(receptance, rep_key, _audit_core_value, weight, a, b)'),
            ("v = rearrange(value, 't b (h n) -> b t h n', h=H)",
             '_audit_correction_value = _audit_source.to(torch.bfloat16) if _audit_source is not None else value\n'
             "    v = rearrange(_audit_correction_value, 't b (h n) -> b t h n', h=H)"),
        ])
        modified_forward(layer,[
            ('hidden_states = hidden_states[0]', 'hidden_states = hidden_states[0].to(torch.bfloat16)'),
            ('(attention_output[0], bias)', '(attention_output[0].to(torch.bfloat16), bias)'),
        ])
    final=model.decoder.final_layernorm
    original_norm=final.forward
    def hidden_only(carrier):
        hidden=original_norm(carrier[0].to(torch.bfloat16))
        return torch.cat((hidden.unsqueeze(0),carrier[1:]),dim=0)
    final.forward=hidden_only
    # RWKV7Model strips the carrier before calling the output layer.
    model.output_layer.register_forward_pre_hook(lambda module,args:(args[0].to(torch.bfloat16),*args[1:]))
