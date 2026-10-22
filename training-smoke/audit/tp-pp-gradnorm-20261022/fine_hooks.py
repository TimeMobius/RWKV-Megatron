"""Capture first-microbatch recomputation boundaries without changing values."""
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.tensor_parallel import mappings
from megatron.core.utils import unwrap_model


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return [cpu(x) for x in value]
    return None


def install(models, out):
    handles, saved, pending = [], set(), {}
    active = [None]
    rank = dist.get_rank()
    original_reduce = mappings._reduce_scatter_along_first_dim

    def reduce_scatter(input_, group, *args, **kwargs):
        label = active[0]
        before = cpu(input_) if label is not None else None
        result = original_reduce(input_, group, *args, **kwargs)
        if label is not None:
            torch.save(dict(input=before, output=cpu(result)),
                       out / f"fine-{label}-collective-rank{rank}.pt")
        return result

    mappings._reduce_scatter_along_first_dim = reduce_scatter
    paths = ["input_layernorm", "self_attention.linear_receptance", "self_attention.linear_key",
             "self_attention.linear_value", "self_attention.core_attention", "self_attention.group_norm",
             "self_attention.bottleneck_gate", "self_attention.linear_proj", "self_attention",
             "pre_mlp_layernorm", "mlp.linear_up", "mlp.linear_down", "mlp"]
    for model in models:
        for layer in unwrap_model(model).decoder.layers:
            number = layer.layer_number - 1
            if number not in (0, 1, 2):
                continue
            modules = dict(layer.named_modules())
            for name in paths + [""]:
                module = modules[name]
                label = f"layer{number}-{name or 'layer'}"
                row_linear = name in ("self_attention.linear_proj", "mlp.linear_down")

                def pre(module, inputs, label=label, row_linear=row_linear):
                    if not torch.is_grad_enabled() or label in saved or label in pending:
                        return
                    pending[label] = cpu(inputs)
                    if row_linear:
                        active[0] = label

                def post(module, inputs, output, label=label, row_linear=row_linear):
                    if label not in pending:
                        return
                    torch.save(dict(inputs=pending.pop(label), output=cpu(output)),
                               out / f"fine-{label}-rank{rank}.pt")
                    saved.add(label)
                    if row_linear:
                        active[0] = None

                handles.append(module.register_forward_pre_hook(pre))
                handles.append(module.register_forward_hook(post))

    def cleanup():
        mappings._reduce_scatter_along_first_dim = original_reduce
        for handle in handles:
            handle.remove()

    return cleanup
