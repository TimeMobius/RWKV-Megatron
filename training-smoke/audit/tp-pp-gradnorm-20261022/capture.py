"""Read-only TP/PP diagnostics on the real training trajectory (DP=CP=1)."""
import hashlib
import json
import os
import re
import runpy
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "3rdparty/Megatron-LM-0.11.0")]
import megatron.training.training as training
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer.module import param_is_not_shared
from megatron.core.utils import unwrap_model

OUT = Path(os.environ["TPPP_DIAG_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
ORIGINAL = training.train_step
COUNT = 0
SELECTED = {1, 24, 25, 26, 27}
REPLAY_BATCHES = []


def named(model):
    chunk = unwrap_model(model)
    for name, param in chunk.named_parameters():
        match = re.search(r"decoder\.layers\.(\d+)\.", name)
        if match:
            number = chunk.decoder.layers[int(match[1])].layer_number - 1
            name = name[:match.start(1)] + str(number) + name[match.end(1):]
        yield name, param


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def gradient_stats(models, step, dump_small=False):
    rows = []
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    small = {}
    for model in models:
        for name, param in named(model):
            grad = param.main_grad
            flat = grad.reshape(-1)
            square = torch.zeros((), device=grad.device, dtype=torch.float64)
            for offset in range(0, flat.numel(), 1048576):
                piece = flat[offset:offset + 1048576].double()
                square += torch.dot(piece, piece)
            included = param_is_not_shared(param) and param_is_not_tensor_parallel_duplicate(param)
            if included:
                total += square
            rows.append(dict(name=name, shape=list(param.shape), squared_norm=square.item(),
                             included=included, tp_sharded=bool(getattr(param, "tensor_model_parallel", False)),
                             partition_dim=getattr(param, "partition_dim", None)))
            if (step == 26 or dump_small) and param.numel() <= 2_000_000:
                small[name] = grad.detach().cpu().clone()
    local = total.item()
    dist.all_reduce(total)
    if small:
        torch.save(small, OUT / f"small-grads-step{step}-rank{dist.get_rank()}.pt")
    return dict(local_squared_norm=local, independently_recomputed_norm=total.sqrt().item(), parameters=rows)


def train_step(*args, **kwargs):
    global COUNT
    COUNT += 1
    step = COUNT
    models, opt = args[2], args[3]
    assert ps.get_data_parallel_world_size(with_context_parallel=True) == 1
    replay = os.environ.get("TPPP_REPLAY_WEIGHTS")
    if replay and step == 1:
        sources = [torch.load(path, mmap=True, weights_only=True)
                   for path in sorted(Path(replay).glob("weights26-rank*.pt"))]
        by_name = {}
        for source in sources:
            for name, value in source["parameters"].items():
                by_name.setdefault(name, []).append((source["tp"], value, source["metadata"][name]))
        with torch.no_grad():
            for model in models:
                for name, param in named(model):
                    pieces = sorted(by_name[name], key=lambda row: row[0])
                    if pieces[0][2]["tp_sharded"]:
                        dim = pieces[0][2]["partition_dim"]
                        full = torch.cat([row[1] for row in pieces], dim=dim)
                        value = full.chunk(ps.get_tensor_model_parallel_world_size(), dim=dim)[ps.get_tensor_model_parallel_rank()]
                    else:
                        value = pieces[0][1]
                    assert value.shape == param.shape, (name, value.shape, param.shape)
                    param.copy_(value)
        del sources, by_name
    # Snapshot the exact BF16 weights consumed by step26, before any update.
    if step == 26 and os.environ.get("TPPP_SAVE_WEIGHTS", "1") == "1":
        values, metadata = {}, {}
        for model in models:
            for name, param in named(model):
                values[name] = param.detach().cpu().clone()
                metadata[name] = dict(tp_sharded=bool(getattr(param, "tensor_model_parallel", False)),
                                      partition_dim=getattr(param, "partition_dim", None))
        torch.save(dict(parameters=values, metadata=metadata, tp=ps.get_tensor_model_parallel_rank()),
                   OUT / f"weights26-rank{dist.get_rank()}.pt")
        del values, metadata
        dist.barrier()
    globals_ = args[0].__globals__
    original_batch = globals_["get_batch"]
    batches = []
    replay_skip_done = False
    batch_index = 0

    def batch(*a, **kw):
        nonlocal replay_skip_done, batch_index
        if replay and step == 1 and not replay_skip_done:
            for _ in range(50):
                original_batch(*a, **kw)
            replay_skip_done = True
        if replay and step > 1:
            result = [x.cuda() if x is not None else None for x in REPLAY_BATCHES[batch_index]]
        else:
            result = original_batch(*a, **kw)
            if replay:
                REPLAY_BATCHES.append([x.detach().cpu().clone() if x is not None else None for x in result])
        if (os.environ.get("TPPP_ZERO_FIRST_LOSS") == "after-first" and step > 1
                and batch_index == 0 and result[2] is not None):
            result = list(result)
            result[2] = result[2].clone()
            result[2][:, 0] = 0
        batch_index += 1
        if step in SELECTED or replay:
            batches.append({key: digest(value) for key, value in zip(
                ("tokens", "labels", "mask", "attention", "positions"), result) if value is not None})
        return result

    original_step = opt.step
    record = dict(step=step, rank=dist.get_rank(), tp=ps.get_tensor_model_parallel_rank(),
                  pp=ps.get_pipeline_model_parallel_rank())

    def optimizer_step(*a, **kw):
        if step in SELECTED or replay:
            record.update(gradient_stats(models, step, dump_small=bool(replay)))
        result = (True, 0.0, None) if replay else original_step(*a, **kw)
        if step in SELECTED or replay:
            record["optimizer_reported_norm"] = float(result[1])
            record["batches"] = batches
            (OUT / f"stats-step{step}-rank{dist.get_rank()}.json").write_text(json.dumps(record, indent=2) + "\n")
        return result

    globals_["get_batch"] = batch
    opt.step = optimizer_step
    handles = []
    if replay and os.environ.get("TPPP_CAPTURE_LAYERS"):
        selected_layers = {int(x) for x in os.environ["TPPP_CAPTURE_LAYERS"].split(",")}
        for model in models:
            for layer in unwrap_model(model).decoder.layers:
                number = layer.layer_number - 1
                if number not in selected_layers:
                    continue
                attention = layer.self_attention
                for label, module in [("key", attention.linear_key), ("receptance", attention.linear_receptance),
                                      ("value", attention.linear_value), ("wkv", attention.core_attention),
                                      ("groupnorm", attention.group_norm)]:
                    calls = [0]

                    def hook(module, inputs, output, label=label, number=number, calls=calls):
                        if not torch.is_grad_enabled():
                            return
                        index = calls[0]
                        calls[0] += 1
                        prefix = OUT / f"layer{number}-{label}-mb{index}-rank{dist.get_rank()}"
                        if step > 1:
                            prefix = Path(str(prefix) + f"-step{step}")
                        value = output[0] if isinstance(output, tuple) else output
                        torch.save(dict(inputs=[x.detach().cpu().clone() for x in inputs if isinstance(x, torch.Tensor)],
                                        output=value.detach().cpu().clone()), str(prefix) + ".pt")
                        if value.requires_grad:
                            def save_grad(grad):
                                torch.save(grad.detach().cpu(), str(prefix) + "-grad.pt")
                            value.register_hook(save_grad)

                    handles.append(module.register_forward_hook(hook))
    original_forward = args[0]
    forward_calls = 0

    def forward(*a, **kw):
        nonlocal forward_calls
        output, loss = original_forward(*a, **kw)
        if replay and ps.is_pipeline_last_stage() and isinstance(output, torch.Tensor):
            torch.save(output.detach().cpu(), OUT / f"token-loss-step{step}-mb{forward_calls}-rank{dist.get_rank()}.pt")
        forward_calls += 1
        return output, loss

    call_args = list(args)
    call_args[0] = forward
    fine_cleanup = None
    if os.environ.get("TPPP_FINE_TRACE"):
        from fine_hooks import install
        fine_cleanup = install(models, OUT)
    try:
        return ORIGINAL(*call_args, **kwargs)
    finally:
        globals_["get_batch"] = original_batch
        opt.step = original_step
        for handle in handles:
            handle.remove()
        if fine_cleanup is not None:
            fine_cleanup()


training.train_step = train_step
runpy.run_path(str(ROOT / "pretrain_rwkv7.py"), run_name="__main__")
