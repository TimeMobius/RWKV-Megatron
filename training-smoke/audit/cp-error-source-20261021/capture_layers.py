"""Production model diagnostics; first microbatch only, no optimizer updates."""
import os
import runpy
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import megatron.training.training as training
from megatron.core import parallel_state as ps
from megatron.core.utils import unwrap_model

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ['CP_DIAG_OUTPUT'])
ORIGINAL = training.train_step


def diagnostic_step(*args, **kwargs):
    models = args[2]
    optimizer = args[3]
    cp, tp, pp = ps.get_context_parallel_rank(), ps.get_tensor_model_parallel_rank(), ps.get_pipeline_model_parallel_rank()
    trace, handles = {}, []

    def save(key, tensor, full=False):
        if key not in trace:
            t = tensor.detach()
            if not full:
                t = t[..., :128]
            trace[key] = t.cpu().clone()

    for model in models:
        for layer in unwrap_model(model).decoder.layers:
            number = layer.layer_number - 1
            attn = layer.self_attention
            def hook(module, inputs, output, number=number):
                save(f'{number}.layer', output[0][0])
            handles.append(layer.register_forward_hook(hook))
            def norm(module, inputs, output, number=number):
                save(f'{number}.norm', output)
            handles.append(layer.input_layernorm.register_forward_hook(norm))
            def attention(module, inputs, output, number=number):
                save(f'{number}.attention', output[0][0])
            handles.append(attn.register_forward_hook(attention))
            def core(module, inputs, output, number=number):
                save(f'{number}.wkv', output)
                for name, tensor in zip(('r', 'k', 'v', 'w', 'a', 'b'), inputs):
                    save(f'{number}.input_{name}', tensor)
            handles.append(attn.core_attention.register_forward_hook(core))
            if number == 0:
                for name in ('linear_key', 'linear_receptance', 'bottleneck_learn'):
                    module = getattr(attn, name)
                    def linear_hook(module, inputs, output, name=name):
                        save(f'gemm.{name}.input', inputs[0], True)
                        save(f'gemm.{name}.output', output[0] if isinstance(output, tuple) else output, True)
                        weight = getattr(module, 'weight', None)
                        if weight is None:
                            weight = module.weight_down
                        save(f'gemm.{name}.weight', weight, True)
                    handles.append(module.register_forward_hook(linear_hook))
    original_step = optimizer.step
    def skip_update(*a, **kw):
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save(trace, OUT / f'trace-tp{tp}-pp{pp}-cp{cp}.pt')
        dist.barrier()
        print(f'CP-DIAG saved tp{tp} pp{pp} cp{cp}', flush=True)
        return True, 0.0, None
    optimizer.step = skip_update
    try:
        return ORIGINAL(*args, **kwargs)
    finally:
        optimizer.step = original_step
        for handle in handles:
            handle.remove()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    training.train_step = diagnostic_step
    sys.argv[0] = str(ROOT / 'pretrain_rwkv7.py')
    runpy.run_path(str(ROOT / 'pretrain_rwkv7.py'), run_name='__main__')
