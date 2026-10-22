"""Observe production training and optionally replay every WKV with identical inputs.

No replacement of model, optimizer, gradient reductions or checkpoint operations.
WKV replay runs once per layer with detached real inputs and a deterministic
incoming derivative, separately from the end-to-end training backward.
"""
import hashlib
import importlib
import json
import os
import runpy
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import megatron.training.training as training
from megatron.core import parallel_state as ps
from megatron.core.utils import unwrap_model

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ['CP_VALIDATION_OUT'])
OUT.mkdir(parents=True, exist_ok=True)
ORIGINAL_STEP = training.train_step
ORIGINAL_EVAL = training.evaluate
EVENTS = []
WKV = {}
OPERANDS = Counter()
DISPATCH = {}
ROUTING = []
WKV_DTYPES = set()
ACTIVE = None
ITERATION = None


def fingerprint(t):
    t = t.detach().contiguous().cpu()
    return dict(shape=list(t.shape), dtype=str(t.dtype),
                sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy()).hexdigest())


def nested(value):
    if isinstance(value, torch.Tensor):
        return fingerprint(value)
    if isinstance(value, dict):
        return {str(k): nested(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [nested(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value))


def flush():
    (OUT/f'rank{dist.get_rank()}.json').write_text(json.dumps(dict(
        events=EVENTS, wkv=WKV, fp8_dispatch=DISPATCH, gemm_operands=dict(OPERANDS),
        optimizer_routing=ROUTING, wkv_dtypes=sorted(WKV_DTYPES)), indent=2)+'\n')


def metric(actual, reference):
    a, r = actual.detach().float(), reference.detach().float()
    d = a-r
    return dict(max_abs=float(d.abs().max()), relative_l2=float(d.norm()/r.norm().clamp_min(1e-30)),
                exact=bool(torch.equal(a, r)), finite=bool(a.isfinite().all() and r.isfinite().all()),
                elements=a.numel())


def replay_wkv(module, inputs, output, layer):
    if layer in WKV:
        return
    cp = ps.get_context_parallel_world_size()
    rank = ps.get_context_parallel_rank()
    group = ps.get_context_parallel_group()
    length = inputs[0].shape[0]
    full = []
    for x in inputs:
        shards = [torch.empty_like(x) for _ in range(cp)]
        dist.all_gather(shards, x.detach().contiguous(), group=group)
        full.append(torch.cat(shards, dim=0).requires_grad_())
    # BF16-exact identical derivative, independent of partition and RNG state.
    go = torch.sin(torch.arange(full[0].numel(), device=inputs[0].device, dtype=torch.float32)*0.013)
    go = go.reshape(full[0].shape).to(inputs[0].dtype)
    impl = importlib.import_module(type(module).__module__)
    original_size = impl.get_context_parallel_world_size
    with torch.enable_grad():
        try:
            impl.get_context_parallel_world_size = lambda: 1
            ref = type(module).forward(module, *full)
            refg = torch.autograd.grad(ref, full, go)
        finally:
            impl.get_context_parallel_world_size = original_size
        local = [x.detach().clone().requires_grad_() for x in inputs]
        actual = type(module).forward(module, *local)
        local_go = go.narrow(0, rank*length, length).contiguous()
        grads = torch.autograd.grad(actual, local, local_go)
    sl = slice(rank*length, (rank+1)*length)
    row = dict(output=metric(actual, ref[sl]), gradients={
        name: metric(g, rg[sl]) for name, g, rg in zip(('r','k','v','w','a','b'), grads, refg)},
        local_shape=list(inputs[0].shape), heads=module.num_heads_local,
        incoming_derivative='sin(global_element_index * 0.013), cast to BF16')
    values = [row['output'], *row['gradients'].values()]
    row['passed'] = all(v['finite'] for v in values) and row['output']['relative_l2'] < 0.001 and all(
        v['relative_l2'] < 0.01 for v in row['gradients'].values())
    WKV[layer] = row
    flush()
    assert all(v['finite'] for v in values), (layer, row)
    if os.environ.get('CP_VALIDATION_WKV_ENFORCE', '1') == '1':
        assert row['passed'], (layer, row)
    print(f'WKV-REPLAY rank={dist.get_rank()} layer={layer} passed', flush=True)


def state(models, optimizer, scheduler, label):
    rows = {}
    names = {}
    for i, model in enumerate(unwrap_model(models)):
        for name, p in model.named_parameters():
            key = f'chunk{i}.{name}'
            names[p] = key
            rows['model.'+key] = fingerprint(p)
    for i, opt in enumerate(getattr(optimizer, 'chained_optimizers', [optimizer])):
        if hasattr(opt, 'gbuf_ranges'):
            for ranges in opt.gbuf_ranges:
                for buckets in ranges.values():
                    for bucket in buckets:
                        for p in bucket['param_map']:
                            for k, v in opt._get_main_param_and_optimizer_states(p).items():
                                rows[f'optimizer{i}.{names[p]}.{k}'] = nested(v)
        rows[f'optimizer{i}.groups'] = nested([{k:v for k,v in g.items() if k != 'params'}
                                               for g in opt.optimizer.param_groups])
    rows['scheduler'] = nested(scheduler.state_dict())
    (OUT/f'{label}-rank{dist.get_rank()}.json').write_text(json.dumps(rows, indent=2)+'\n')


def train_step(*args, **kwargs):
    global ACTIVE, ITERATION
    ITERATION = kwargs['iteration'] + 1
    models, optimizer, scheduler = args[2:5]
    forward = args[0]
    globals_ = forward.__globals__
    get_batch = globals_['get_batch']
    ACTIVE = dict(kind='train', iteration=ITERATION, batches=[], token_loss=[],
                  consumed_train_samples=training.get_args().consumed_train_samples)
    def batch(*a, **kw):
        value = get_batch(*a, **kw)
        ACTIVE['batches'].append(nested(value))
        return value
    def forward_step(*a, **kw):
        output, loss = forward(*a, **kw)
        if output.requires_grad:
            ACTIVE['token_loss'].append(fingerprint(output))
        return output, loss
    hooks = []
    chunks = unwrap_model(models)
    if ITERATION == 1 and os.environ.get('CP_VALIDATION_WKV') == '1':
        for chunk in chunks:
            for layer in chunk.decoder.layers:
                number = str(layer.layer_number - 1)
                def hook(m, x, y, number=number):
                    replay_wkv(m, x, y, number)
                hooks.append(layer.self_attention.core_attention.register_forward_hook(hook))
    globals_['get_batch'] = batch
    step_args = list(args)
    step_args[0] = forward_step
    torch.cuda.reset_peak_memory_stats()
    try:
        if os.environ.get('CP_VALIDATION_STATE') == '1' and ITERATION == 3:
            state(models, optimizer, scheduler, 'before3')
        result = ORIGINAL_STEP(*step_args, **kwargs)
        ACTIVE.update(losses={k:float(v) for k,v in result[0].items()}, skipped=int(result[1]),
                      grad_norm=float(result[5]) if result[5] is not None else None,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      chunks=[dict(pre_process=c.pre_process, post_process=c.post_process,
                                   layers=[layer.layer_number for layer in c.decoder.layers]) for c in chunks])
        EVENTS.append(ACTIVE)
        flush()
        if os.environ.get('CP_VALIDATION_STATE') == '1' and ITERATION == 4:
            state(models, optimizer, scheduler, 'after4')
        return result
    finally:
        globals_['get_batch'] = get_batch
        for h in hooks:
            h.remove()
        ACTIVE = None


def evaluate(*args, **kwargs):
    global ACTIVE
    ACTIVE = dict(kind='eval', iteration=ITERATION)
    try:
        result = ORIGINAL_EVAL(*args, **kwargs)
        ACTIVE['losses'] = {k:float(v) for k,v in result[0].items()}
        EVENTS.append(ACTIVE)
        flush()
        return result
    finally:
        ACTIVE = None


def install_dispatch():
    import megatron.core.optimizer as optimizers
    from megatron.core.optimizer.layer_wise_optimizer import is_managed_by_layer_wise_optimizer
    original_groups = optimizers._get_param_groups
    def groups(chunks, config, *a, **kw):
        result = original_groups(chunks, config, *a, **kw)
        if config.optimizer == 'muon':
            names = {p: f'{i}.{name}' for i,c in enumerate(chunks) for name,p in c.named_parameters()}
            for group in result:
                selected = group.get('optimizer', config.optimizer)
                for p in group['params']:
                    expected = 'muon' if is_managed_by_layer_wise_optimizer(p) else 'adam'
                    assert selected == expected, (names[p], selected, expected)
                    ROUTING.append(dict(name=names[p], shape=list(p.shape), optimizer=selected))
        return result
    optimizers._get_param_groups = groups
    from megatron.core.extensions.transformer_engine import TELinear
    original = TELinear.forward
    def linear(self, *a, **kw):
        value = original(self, *a, **kw)
        key = getattr(self, '_cp_validation_name', str(id(self)))
        DISPATCH[key] = dict(fp8=bool(self.fp8), recipe=type(self.fp8_meta.get('recipe')).__name__)
        return value
    TELinear.forward = linear
    from rwkv_megatron.rwkv7_model import RWKV7Model
    init = RWKV7Model.__init__
    def model_init(self, *a, **kw):
        init(self, *a, **kw)
        for name, m in self.named_modules():
            if isinstance(m, TELinear):
                m._cp_validation_name = f'{self.vp_stage}.{name}'
        for layer in self.decoder.layers:
            def dtype_hook(m, inputs):
                WKV_DTYPES.update(str(t.dtype) for t in inputs)
                assert all(t.dtype == torch.bfloat16 for t in inputs)
            layer.self_attention.core_attention.register_forward_pre_hook(dtype_hook)
    RWKV7Model.__init__ = model_init
    gemm_module = importlib.import_module('transformer_engine.pytorch.module.linear')
    gemm = gemm_module.general_gemm
    def observed_gemm(a, b, *args, **kw):
        OPERANDS[f'{type(a).__name__} x {type(b).__name__}: grad={bool(kw.get("grad"))}'] += 1
        return gemm(a, b, *args, **kw)
    gemm_module.general_gemm = observed_gemm


if __name__ == '__main__':
    training.train_step = train_step
    training.evaluate = evaluate
    install_dispatch()
    runpy.run_path(str(ROOT/'pretrain_rwkv7.py'), run_name='__main__')
