"""Process-local unused-FLA candidates; production parameters/code untouched."""
import inspect
import textwrap

from fla.modules.layernorm import group_norm
from fla.ops.rwkv7.gate_output_correction import gate_output_correction
from v0_fp32 import modified_forward


def correction(o,r,k,rho,v,g,head_size):
    shape=o.shape
    shaped=lambda t:t.reshape(1,-1,shape[-1]//head_size,head_size)
    return gate_output_correction(o.reshape(1,-1,shape[-1]),shaped(r),shaped(k),
        rho.reshape(-1,head_size),shaped(v),g.reshape(1,-1,shape[-1])).reshape(shape)


def install(model,mode):
    assert model.config.tensor_model_parallel_size==1 and model.config.pipeline_model_parallel_size==1
    assert not model.config.sequence_parallel
    for layer in model.decoder.layers:
        att=layer.self_attention
        if mode in ('norm','norm-correction'):
            module=att.group_norm
            module.forward=lambda x,module=module:group_norm(x,module.weight,module.bias,
                                                          eps=module.eps,num_groups=module.num_groups)
            if mode=='norm':
                # FLA returns a view from a custom autograd function, which
                # cannot be modified by the current in-place correction add.
                modified_forward(att,[('output += ut','output = output + ut')])
        if mode in ('correction','norm-correction'):
            att._audit_fla_correction=correction
            source=textwrap.dedent(inspect.getsource(type(att).forward))
            start=source.index('    # B = output.shape[1]')
            end=source.index('    with te.fp8_autocast',start)
            before=source[start:end]
            modified_forward(att,[(before,'    # Output correction and gate run inside the fused call below.\n'),
                ('self.linear_proj(output * gate)',
                 'self.linear_proj(self._audit_fla_correction(output, receptance, rep_key, self.rho, value, gate, self.head_size))')])
        if mode=='wkv-save':
            modified_forward(att.core_attention,[
                ('safe_gate=True, chunk_size=64)',
                 'safe_gate=True, chunk_size=64, lower_bound=-0.6065306597126334, disable_recompute=True)'),
            ])
