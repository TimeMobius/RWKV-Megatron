"""Isolated LayerNorm accuracy and eager speed evaluation; no production edits."""
import gc
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine
from megatron.core import parallel_state as ps
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars
from rwkv_megatron.rwkv7_model import RWKV7Model
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
import distributed as audit

OUT = Path(__file__).resolve().parent
MODEL = '/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth'


def metric(value, reference):
    actual, expected = value.detach().double(), reference.detach().double()
    diff = actual - expected
    return {'relative_l2': (diff.norm() / expected.norm().clamp_min(1e-30)).item(),
            'max_abs': diff.abs().max().item(),
            'finite': bool(torch.isfinite(actual).all())}


def build(backend, config, weight, bias):
    if backend == 'torch':
        module = torch.nn.LayerNorm(2560, eps=1e-5, device='cuda', dtype=torch.bfloat16)
    elif backend == 'apex':
        module = FusedLayerNorm(config, 2560, 1e-5).cuda().to(torch.bfloat16)
    else:
        module = TENorm(config, 2560, 1e-5)
    with torch.no_grad():
        module.weight.copy_(weight)
        module.bias.copy_(bias)
    return module


def elapsed(action, iterations=40):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        action()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    torch.cuda.set_device(0)
    dist.init_process_group('nccl')
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    global_vars._GLOBAL_ARGS = SimpleNamespace(transformer_impl='local')
    audit.FULL = True
    audit.DIMS = {'layers': 32, 'hidden': 2560, 'heads': 40, 'ffn': 10240,
                  'vocab': 65536, 'sequence': 128, 'batch': 1, 'microbatches': 1}
    cfg = audit.config_for(1, 1, False, False, fp32=False)
    state = torch.load(MODEL, map_location='cpu', weights_only=True, mmap=True)
    tokens = torch.load(OUT / 'minipile-tokens.pth', weights_only=True).cuda()[:, :-1]
    model = RWKV7Model(cfg, transformer_layer_spec=get_rwkv7_layer_spec(), vocab_size=65536,
        max_sequence_length=128, pre_process=True, post_process=True, parallel_output=False,
        share_embeddings_and_output_weights=False, position_embedding_type='none').cuda().to(torch.bfloat16)
    model.from_rwkv_lm(state, False)
    captures = {}
    handles = []
    modules = {'ln0': model.pre_norm, '0.ln1': model.decoder.layers[0].input_layernorm,
        '0.ln2': model.decoder.layers[0].pre_mlp_layernorm,
        '15.ln1': model.decoder.layers[15].input_layernorm,
        '31.ln2': model.decoder.layers[31].pre_mlp_layernorm,
        'ln_out': model.decoder.final_layernorm}
    production_types = {name: type(module).__module__ + '.' + type(module).__name__
                        for name, module in modules.items()}
    for name, module in modules.items():
        def capture(module, args, name=name):
            x = args[0]
            if x.ndim == 4:
                x = x[0]  # hidden only; discarded value-carrier norm is not part of this test.
            captures[name] = {'input': x.detach().cpu().contiguous(),
                              'weight': module.weight.detach().cpu().clone(),
                              'bias': module.bias.detach().cpu().clone()}
        handles.append(module.register_forward_pre_hook(capture))
    with torch.no_grad():
        logits = model(tokens, torch.arange(128, device='cuda').expand_as(tokens), None)
    for handle in handles:
        handle.remove()
    del logits, modules, model, state
    gc.collect()
    torch.cuda.empty_cache()
    torch.save(captures, OUT / 'layernorm-inputs-current.pth')
    accuracy = {}
    for index, (name, data) in enumerate(captures.items()):
        torch.manual_seed(20261009 + index)
        x0, w0, b0 = [data[key].cuda() for key in ('input', 'weight', 'bias')]
        dy = torch.randn_like(x0)
        xr, wr, br = [t.double().requires_grad_() for t in (x0, w0, b0)]
        yr = F.layer_norm(xr, (2560,), wr, br, 1e-5)
        gr = torch.autograd.grad(yr, (xr, wr, br), dy.double())
        accuracy[name] = {}
        ys = {}
        for backend in ('torch', 'apex', 'te'):
            module = build(backend, cfg, w0, b0)
            x = x0.clone().requires_grad_()
            y = module(x)
            grads = torch.autograd.grad(y, (x, module.weight, module.bias), dy)
            values = {'output': y, 'input_gradient': grads[0],
                      'weight_gradient': grads[1], 'bias_gradient': grads[2]}
            refs = {'output': yr, 'input_gradient': gr[0],
                    'weight_gradient': gr[1], 'bias_gradient': gr[2]}
            accuracy[name][backend] = {
                key: {'vs_fp64': metric(value, refs[key]),
                      'vs_fp64_rounded_bf16': metric(value, refs[key].to(torch.bfloat16))}
                for key, value in values.items()}
            ys[backend] = y.detach()
        accuracy[name]['forward_pairs'] = {
            'te_vs_torch': metric(ys['te'], ys['torch']),
            'apex_vs_torch': metric(ys['apex'], ys['torch']),
            'te_vs_apex': metric(ys['te'], ys['apex'])}
        print('ACCURACY', name,
              {b: accuracy[name][b]['output']['vs_fp64']['relative_l2'] for b in ('torch', 'apex', 'te')},
              flush=True)
    data = captures['ln0']
    w, b = data['weight'].cuda(), data['bias'].cuda()
    benchmarks = []
    for rows, batch in ((128, 1), (512, 1), (4096, 1), (4096, 4)):
        base = data['input'].reshape(-1, 2560)
        x0 = base.repeat((rows * batch + 127) // 128, 1)[:rows * batch].reshape(rows, batch, 2560).cuda()
        dy = torch.randn_like(x0)
        modules = {b: build(b, cfg, w, data['bias'].cuda()) for b in ('torch', 'apex', 'te')}
        inputs = {b: x0.clone().requires_grad_() for b in modules}
        results = {b: {'forward_us': [], 'forward_backward_us': []} for b in modules}
        for round_index in range(5):
            order = list(modules)[round_index % 3:] + list(modules)[:round_index % 3]
            for backend in order:
                module, x = modules[backend], inputs[backend]
                def fwd():
                    with torch.no_grad():
                        module(x)
                def train():
                    module.zero_grad(set_to_none=True)
                    x.grad = None
                    module(x).backward(dy)
                for _ in range(10):
                    fwd()
                    train()
                results[backend]['forward_us'].append(elapsed(fwd))
                results[backend]['forward_backward_us'].append(elapsed(train))
        results = {b: {key: {'median': statistics.median(v), 'trials': v}
                       for key, v in values.items()} for b, values in results.items()}
        benchmarks.append({'shape': [rows, batch, 2560], 'results': results})
        print('SPEED', [rows, batch, 2560],
              {b: {key: round(v['median'], 2) for key, v in values.items()}
               for b, values in results.items()}, flush=True)
        del modules, inputs
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'te': transformer_engine.__version__, 'dtype': 'bfloat16', 'hidden': 2560,
        'eps': 1e-5, 'persist_layer_norm': False, 'sequence_parallel': False,
        'zero_centered_gamma': False, 'production_types': production_types,
        'inputs': 'Fresh current-production 3B forward, fixed MiniPile prefix128; checkpoint BF16 affine weights.',
        'gradient_probe': 'Identical fixed random BF16 output gradient; isolated operator, not full-model CE backward.',
        'reference': 'FP64 native LayerNorm/autograd on the identical quantized BF16 input, affine weights and output gradient.',
        'benchmark': 'Eager CUDA events; 5 rotated-order trials of40 iterations,10 warmup; repeated ln0 data; no graphs.',
        'accuracy': accuracy, 'benchmarks': benchmarks,
        'limitations': ['Single GPU H800, not end-to-end TPS or convergence.',
                        'Only hidden2560 nonpersistent kernels and BF16 affine parameters.',
                        'Sequence parallel/distributed gradient reduction not exercised.',
                        'Forward-only uses no_grad; forward_backward includes eager/autograd dispatch overhead.']}
    (OUT / 'layernorm-evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
