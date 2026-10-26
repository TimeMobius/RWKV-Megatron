"""Trace production training without changing inputs or gradient normalization.

DIAG_ROW_FP32 is an experimental row-GEMM output control, never a production
setting. Tensor clones stay on GPU until the step completes to limit hook sync.
"""

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import torch

import megatron.training.training as training
from megatron.core import parallel_state as ps
from megatron.core.pipeline_parallel import schedules
from megatron.core.tensor_parallel.layers import RowParallelLinear

ROOT = Path(__file__).resolve().parents[1]
CTX = None
TRAIN_STEP = training.train_step
BACKWARD_STEP = schedules.backward_step


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(item) for item in value]
    return value


def unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    return None


def trace_backward(*args, **kwargs):
    result = BACKWARD_STEP(*args, **kwargs)
    if os.environ.get("DIAG_TRACE_LEVEL") == "sync":
        torch.cuda.synchronize()
    if CTX is not None:
        CTX["accum"].append(
            dict(
                key=CTX["backward_key"],
                grads={
                    name: p.main_grad.detach().clone()
                    for name, p in CTX["model"].named_parameters()
                },
            )
        )
    return result


def trace_step(forward_func, iterator, model, optimizer, *args, **kwargs):
    global CTX
    if os.environ.get("DIAG_REWARM_CHECKPOINT") and kwargs["iteration"] == 2:
        # Compare direct restart with two warm training steps, then restore the
        # very same historical model/Adam/RNG before the measured step3.
        from megatron.training import get_args
        from megatron.training.checkpointing import load_checkpoint

        config = get_args()
        config.diag_rewarm_load = os.environ["DIAG_REWARM_CHECKPOINT"]
        restored_iteration, _ = load_checkpoint(
            model, optimizer, None, load_arg="diag_rewarm_load"
        )
        del config.diag_rewarm_load
        assert restored_iteration == 2
    level = os.environ.get("DIAG_TRACE_LEVEL", "full")
    if level in ("none", "sync", "single-thread", "eager-mix", "plain", "cold"):
        if level == "single-thread":
            with torch.autograd.set_multithreading_enabled(False):
                result = TRAIN_STEP(
                    forward_func, iterator, model, optimizer, *args, **kwargs
                )
        else:
            result = TRAIN_STEP(
                forward_func, iterator, model, optimizer, *args, **kwargs
            )
        configs = {}
        from triton.runtime.autotuner import Autotuner

        for module_name, module_obj in list(sys.modules.items()):
            if not module_name.startswith("fla.ops.") or module_obj is None:
                continue
            for key, value in vars(module_obj).items():
                # Many recurrence kernels put Heuristics outside Autotuner.
                # Looking only at the outer object misses their tile choices.
                seen = set()
                while not isinstance(value, Autotuner) and hasattr(value, "fn"):
                    if id(value) in seen:
                        break
                    seen.add(id(value))
                    value = value.fn
                if isinstance(value, Autotuner):
                    if value.cache:
                        configs[module_name + "." + key] = {
                            str(k): str(v) for k, v in value.cache.items()
                        }
                    elif len(value.configs) == 1:
                        configs[module_name + "." + key] = {
                            "fixed": str(value.configs[0])
                        }
        path = (
            Path(os.environ["RWKV_DP_AUDIT_DIR"])
            / f'kernels-step{kwargs["iteration"]+1}-rank{torch.distributed.get_rank()}.json'
        )
        path.write_text(json.dumps(configs, indent=2) + "\n")
        return result
    module = unwrap(model[0])
    CTX = dict(
        step=kwargs["iteration"] + 1,
        model=module,
        current=None,
        backward_key=None,
        modules={},
        param_grads={},
        accum=[],
        forwards=[],
    )
    context = CTX
    handles = []

    def root_pre(mod, inputs):
        tokens = inputs[0]
        key = hashlib.sha256(
            tokens.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        context["current"] = key
        context["forwards"].append(key)

    def root_post(mod, inputs, output):
        value = tensor(output)
        if value is not None and value.requires_grad:
            key = context["current"]

            def start_backward(grad):
                context["backward_key"] = key

            value.register_hook(start_backward)

    handles.extend(
        [
            module.register_forward_pre_hook(root_pre),
            module.register_forward_hook(root_post),
        ]
    )
    selected = (
        "input_layernorm",
        "pre_mlp_layernorm",
        "self_attention.linear_key",
        "self_attention.bottleneck_decay",
        "self_attention.core_attention",
        "self_attention.linear_proj",
        "mlp.linear_down",
    )
    for layer in module.decoder.layers:
        global_layer = layer.self_attention.layer_number - 1
        for suffix in selected:
            child = layer.get_submodule(suffix)
            name = f"layer{global_layer}.{suffix}"

            def capture(mod, inputs, output, name=name):
                key = context["current"]
                entry = {}
                context["modules"].setdefault(key, {})[name] = entry
                observed = [
                    ("input", tensor(inputs)),
                    ("output", tensor(output)),
                ]
                if os.environ.get("DIAG_ALL_WKV_INPUTS") == "1" and name.endswith(
                    "core_attention"
                ):
                    observed.extend(
                        (f"operand{index}", value) for index, value in enumerate(inputs)
                    )
                for label, value in observed:
                    if value is None:
                        continue
                    entry[label] = value.detach().clone()
                    if value.requires_grad:

                        def capture_grad(grad, label=label, entry=entry):
                            entry[label + "_grad"] = grad.detach().clone()

                        value.register_hook(capture_grad)

            handles.append(child.register_forward_hook(capture))
    for name, param in module.named_parameters():

        def capture_param(grad, name=name):
            context["param_grads"].setdefault(context["backward_key"], {})[
                name
            ] = grad.detach().clone()

        handles.append(param.register_hook(capture_param))
    try:
        result = TRAIN_STEP(forward_func, iterator, model, optimizer, *args, **kwargs)
        context.pop("model")
        context.update(
            tp=ps.get_tensor_model_parallel_rank(),
            pp=ps.get_pipeline_model_parallel_rank(),
            dp=ps.get_data_parallel_rank(),
        )
        torch.save(
            cpu(context),
            Path(os.environ["RWKV_DP_AUDIT_DIR"])
            / f'trace-step{context["step"]}-rank{torch.distributed.get_rank()}.pt',
        )
        return result
    finally:
        for handle in handles:
            handle.remove()
        CTX = None


if __name__ == "__main__":
    if os.environ.get("DIAG_DGK_LAYOUT"):
        import importlib

        from triton.runtime.autotuner import Autotuner

        dgk_module = importlib.import_module(
            "fla.ops.generalized_delta_rule.dplr.chunk_A_bwd"
        )
        tuner = dgk_module.chunk_dplr_bwd_dgk_kernel
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        use_alt = (
            os.environ["DIAG_DGK_LAYOUT"] == "historical"
            and int(os.environ["RANK"]) == 5
        )
        block = 64 if use_alt else 32
        tuner.configs = [
            next(
                c
                for c in tuner.configs
                if c.kwargs["BK"] == block and c.num_warps == 4 and c.num_stages == 2
            )
        ]
        tuner.cache.clear()
    if os.environ.get("DIAG_DGK_CAPTURE") == "1":
        import importlib

        from triton.runtime.autotuner import Autotuner

        dgk_module = importlib.import_module(
            "fla.ops.generalized_delta_rule.dplr.chunk_A_bwd"
        )
        dgk_tuner = dgk_module.chunk_dplr_bwd_dgk_kernel
        while not isinstance(dgk_tuner, Autotuner):
            dgk_tuner = dgk_tuner.fn
        dgk_run = dgk_tuner.run

        def capture_dgk(*args, **kwargs):
            result = dgk_run(*args, **kwargs)
            if CTX is not None:
                captured = {
                    key: (
                        value.detach().clone()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in kwargs.items()
                    if key not in ("grid", "warmup")
                }
                CTX.setdefault("dgk_kernels", []).append(
                    dict(
                        micro=CTX["backward_key"],
                        operands=captured,
                        launch_config=str(dgk_tuner.best_config),
                    )
                )
            return result

        dgk_tuner.run = capture_dgk
    if os.environ.get("DIAG_BF16_FULL_REDUCTION") == "1":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if os.environ.get("DIAG_TRACE_LEVEL") == "eager-mix":
        import importlib

        mix = importlib.import_module("fla.ops.rwkv7.fused_addcmul")
        mix.addcmul_bwd2 = mix.addcmul_bwd2._torchdynamo_orig_callable
    if os.environ.get("DIAG_ROW_FP32") == "1":
        original_impl, original_forward = (
            RowParallelLinear._forward_impl,
            RowParallelLinear.forward,
        )

        def fp32_impl(self, *args, **kwargs):
            kwargs["output_dtype"] = torch.float32
            return original_impl(self, *args, **kwargs)

        def row_forward(self, value):
            output, bias = original_forward(self, value)
            return output.to(value.dtype), bias

        RowParallelLinear._forward_impl = fp32_impl
        RowParallelLinear.forward = row_forward
    training.train_step = trace_step
    schedules.backward_step = trace_backward
    runpy.run_path(str(ROOT / "tests/run_dp_distill_training.py"), run_name="__main__")
