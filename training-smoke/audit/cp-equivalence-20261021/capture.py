"""Capture production TP/PP/DDP CP gradients before clipping or optimizer updates.

Run via torchrun with a normal --rwkv-config and CP_AUDIT_OUTPUT. Optimizer
updates are skipped, but production forward/backward/finalize_model_grads and
reduce-scatter are unmodified. Reconstruct only the reduced optimizer shards.
"""
import json
import os
import re
import runpy
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import megatron.training.training as training
from megatron.core import parallel_state as ps
from megatron.core.utils import unwrap_model

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ['CP_AUDIT_OUTPUT'])
ORIGINAL = training.train_step


def audit_train_step(*args, **kwargs):
    forward = args[0] if args else kwargs['forward_step_func']
    models = args[2] if len(args) > 2 else kwargs['model']
    optimizer = args[3] if len(args) > 3 else kwargs['optimizer']
    cp = ps.get_context_parallel_rank()
    cp_size = ps.get_context_parallel_world_size()
    tp, pp = ps.get_tensor_model_parallel_rank(), ps.get_pipeline_model_parallel_rank()
    OUT.mkdir(parents=True, exist_ok=True)
    trace = {'batches': [], 'losses': [], 'logits': [], 'head_inputs': []}
    globals_ = forward.__globals__
    get_batch = globals_['get_batch']
    original_step = optimizer.step

    def capture_batch(*a, **kw):
        batch = list(get_batch(*a, **kw))
        tokens, labels, mask, attention, positions = batch
        if mask is not None and os.environ.get('CP_AUDIT_MASK') == 'uneven':
            start = cp * mask.shape[1]
            index = torch.arange(start, start + mask.shape[1], device=mask.device)
            # Deliberately unequal valid counts in the two CP shards, plus a
            # zeroed interval crossing the shard boundary. Same global mask CP1/2.
            length = mask.shape[1] * cp_size
            keep = (index % 11 != 0) & ~((index >= length//2-9) & (index < length//2+13))
            keep &= (index >= length//2) | (index % 3 == 0)
            batch[2] = mask * keep
        trace['batches'].append({key: value.detach().cpu().clone() if value is not None else None
                                 for key, value in zip(('tokens', 'labels', 'mask', 'attention', 'positions'), batch)})
        return batch

    globals_['get_batch'] = capture_batch
    hooks = []
    if unwrap_model(models[-1]).post_process:
        def head_hook(module, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            # Full token coverage and deterministic vocabulary samples. Full CE
            # below covers the complete vocabulary; sampled logits bound disk use.
            cols = torch.linspace(0, logits.shape[-1]-1, 64, device=logits.device).long()
            trace['logits'].append(logits.detach().index_select(-1, cols).cpu())
            head_input = inputs[0].detach()
            if getattr(module, 'sequence_parallel', False):
                shards = [torch.empty_like(head_input) for _ in range(ps.get_tensor_model_parallel_world_size())]
                dist.all_gather(shards, head_input.contiguous(), group=ps.get_tensor_model_parallel_group())
                head_input = torch.cat(shards, dim=0)
            trace['head_inputs'].append(head_input.cpu().clone())
        hooks.append(unwrap_model(models[-1]).output_layer.register_forward_hook(head_hook))

    def capture_forward(*a, **kw):
        output, loss = forward(*a, **kw)
        def capture_loss(tensor, *la, **lk):
            token_loss = tensor.detach().cpu().clone()
            result = loss(tensor, *la, **lk)
            trace['losses'].append({'tokens': token_loss, 'loss_sum_returned': float(result[0].detach()),
                                    'count_returned': int(result[1]),
                                    'report': result[2]['lm loss'].detach().cpu().clone()})
            return result
        return output, capture_loss

    def capture_step(*a, **kw):
        gradients = {}
        shapes = {}
        for model in models:
            names = {param: name for name, param in unwrap_model(model).named_parameters()}
            if os.environ.get('CP_AUDIT_GLOBAL_LAYER_NAMES') == '1':
                chunk = unwrap_model(model)
                def global_layer_name(name):
                    match = re.search(r'decoder\.layers\.(\d+)\.', name)
                    if match is None:
                        return name
                    number = chunk.decoder.layers[int(match[1])].layer_number - 1
                    return name[:match.start(1)] + str(number) + name[match.end(1):]
                names = {p: global_layer_name(n) for p,n in names.items()}
            for buffer in model.buffers:
                group = buffer.data_parallel_group
                size, rank = dist.get_world_size(group), dist.get_rank(group)
                for bucket_id, bucket in enumerate(buffer.buckets):
                    # DDP has already scaled by 1/(DP*CP) and reduce-scattered.
                    # Unowned buffer regions contain stale data; NEVER dump
                    # param.main_grad directly in the distributed optimizer case.
                    if model.ddp_config.use_distributed_optimizer and size > 1:
                        shard = bucket.grad_data.chunk(size)[rank].contiguous()
                        full = torch.empty_like(bucket.grad_data)
                        dist.all_gather_into_tensor(full, shard, group=group)
                    else:
                        full = bucket.grad_data
                    if cp == 0:
                        for param in bucket.params:
                            start, end, index = buffer.param_index_map[param]
                            assert index == bucket_id
                            name = names[param]
                            gradients[name] = full[start-bucket.offset:end-bucket.offset].view(param.shape).detach().cpu().clone()
                            shapes[name] = list(param.shape)
                    del full
        trace['cp'] = cp
        trace['cp_size'] = cp_size
        trace['tp'] = tp
        trace['pp'] = pp
        torch.save(trace, OUT / f'trace-tp{tp}-pp{pp}-cp{cp}.pt')
        if cp == 0:
            torch.save(gradients, OUT / f'grad-tp{tp}-pp{pp}.pt')
            metadata = {'parameters': shapes, 'gradient_elements': sum(g.numel() for g in gradients.values())}
            (OUT / f'meta-tp{tp}-pp{pp}.json').write_text(json.dumps(metadata, indent=2)+'\n')
        dist.barrier()
        print(f'CP-AUDIT captured tp={tp} pp={pp} cp={cp}', flush=True)
        # No clipping, parameter update, Adam state initialization or checkpoint.
        return True, 0.0, None

    optimizer.step = capture_step
    new_args = list(args)
    if new_args:
        new_args[0] = capture_forward
    else:
        kwargs['forward_step_func'] = capture_forward
    try:
        return ORIGINAL(*new_args, **kwargs)
    finally:
        optimizer.step = original_step
        globals_['get_batch'] = get_batch
        for hook in hooks:
            hook.remove()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    training.train_step = audit_train_step
    sys.argv[0] = str(ROOT / 'pretrain_rwkv7.py')
    runpy.run_path(str(ROOT / 'pretrain_rwkv7.py'), run_name='__main__')
