"""Real single-GPU TE dispatch, precision and recomputation checks on native weights."""

import argparse
import gc
import importlib
import json
import os
import statistics
import time
from datetime import timedelta
from types import SimpleNamespace

import distributed as fixture
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine.pytorch as te
from runtime import ROOT, report
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from megatron.core import parallel_state as ps
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model

OUT = ROOT / "training-smoke/audit"


def compare(model, baseline):
    totals = [0.0, 0.0, 0.0]
    count = 0
    first_value = None
    for name, param in model.named_parameters():
        if param.grad is None:
            assert name not in baseline, name
            continue
        actual = param.grad.detach().float().cpu()
        expected = baseline[name].float()
        assert torch.isfinite(actual).all(), name
        delta = actual - expected
        d2, e2 = delta.square().sum().item(), expected.square().sum().item()
        totals[0] += d2
        totals[1] += e2
        totals[2] = max(totals[2], delta.abs().max().item())
        if name == "decoder.layers.0.self_attention.linear_value.weight":
            first_value = (d2 / max(e2, 1e-30)) ** 0.5
        count += 1
    return {
        "parameter_gradient_count": count,
        "relative_l2": (totals[0] / totals[1]) ** 0.5,
        "max_abs": totals[2],
        "first_layer_value_weight_relative_l2": first_value,
    }


def run_case(
    name,
    native,
    tokens,
    labels,
    *,
    recipe=None,
    recompute=False,
    legacy=False,
    first_last=False,
    wgrad=True,
    baseline=None,
    repeats=5,
    capture=False,
    scope="all",
    disable_ffn=None,
):
    FP8GlobalStateManager.reset()
    model_parallel_cuda_manual_seed(1234)
    cfg = fixture.config_for(1, 1, False, recompute)
    cfg.transformer_impl = "transformer_engine"
    cfg.fp8 = "hybrid" if recipe else None
    cfg.fp8_recipe = recipe or "delayed"
    cfg.fp8_wgrad = wgrad
    cfg.rwkv_fp8_scope = scope
    cfg.first_last_layers_bf16 = first_last
    cfg.num_layers_at_start_in_bf16 = 1
    cfg.num_layers_at_end_in_bf16 = 1
    cfg.rwkv_key_fusion = True
    model = (
        RWKV7Model(
            cfg,
            transformer_layer_spec=get_rwkv7_layer_spec(),
            vocab_size=fixture.DIMS["vocab"],
            max_sequence_length=128,
            parallel_output=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type="none",
        )
        .cuda()
        .bfloat16()
    )
    model.from_rwkv_lm(native, False)
    model.train()
    linears = {n: m for n, m in model.named_modules() if isinstance(m, TELinear)}
    assert len(linears) == 6 * cfg.num_layers
    if legacy:
        for module in linears.values():
            forward = module.forward

            def disabled(*args, _forward=forward, **kwargs):
                with te.fp8_autocast(enabled=False):
                    return _forward(*args, **kwargs)

            module.forward = disabled
    if disable_ffn:
        for n, module in linears.items():
            if n.endswith("linear_" + disable_ffn):
                forward = module.forward

                def disabled(*args, _forward=forward, **kwargs):
                    with te.fp8_autocast(enabled=False):
                        return _forward(*args, **kwargs)

                module.forward = disabled
    dtypes = set()

    def wkv_inputs(module, args):
        dtypes.update(str(x.dtype) for x in args)

    handles = [
        layer.self_attention.core_attention.register_forward_pre_hook(wkv_inputs)
        for layer in model.decoder.layers
    ]
    positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
    times, losses = [], []
    # Same weights and identical inputs throughout; warm delayed scales without optimizer steps.
    first = None
    for iteration in range(3 + repeats):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(tokens, positions, None)
        loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        losses.append(loss.item())
        if iteration == 0:
            first = logits.detach().float().cpu()
        if iteration >= 3:
            times.append(elapsed)
    dispatch = {
        n: {
            "fp8": bool(m.fp8),
            "recipe": type(m.fp8_meta.get("recipe")).__name__,
            "quantizers": [type(q).__name__ for q in m.quantizers["scaling_fwd"]],
            "amax_group_present": m.fp8_meta.get("fp8_group") is not None,
        }
        for n, m in linears.items()
    }
    active = sum(x["fp8"] for x in dispatch.values())
    dispatch_groups = {}
    for module_name, metadata in dispatch.items():
        tag = json.dumps(metadata, sort_keys=True)
        group = dispatch_groups.setdefault(tag, dict(metadata, module_names=[]))
        group["module_names"].append(module_name)
    expected_active = (
        0
        if not recipe or legacy
        else ((2 if scope == "ffn" else 6) - bool(disable_ffn))
        * (cfg.num_layers - (2 if first_last else 0))
    )
    assert active == expected_active, (name, active, expected_active)
    assert dtypes == {"torch.bfloat16"}, dtypes
    assert model.output_layer.weight.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()
    final = logits.detach().float().cpu()
    result = {
        "name": name,
        "fp8_recipe": recipe,
        "fp8_wgrad": wgrad,
        "rwkv_fp8_scope": scope,
        "recompute": recompute,
        "first_last_bf16": first_last,
        "losses": losses,
        "median_forward_backward_seconds": statistics.median(times),
        "tokens_per_second": tokens.numel() / statistics.median(times),
        "fp8_linear_count": active,
        "linear_count": len(linears),
        "wkv_dtypes": sorted(dtypes),
        "dispatch_groups": list(dispatch_groups.values()),
    }
    if baseline is not None:
        diff = final - baseline["logits"]
        result["logits"] = {
            "relative_l2": (diff.norm() / baseline["logits"].norm()).item(),
            "max_abs": diff.abs().max().item(),
            "first_pass_relative_l2": (
                (first - baseline["logits"]).norm() / baseline["logits"].norm()
            ).item(),
        }
        result["gradients"] = compare(model, baseline["grads"])
        result["loss_abs_difference"] = abs(losses[-1] - baseline["loss"])
    snapshot = None
    if capture:
        snapshot = {
            "grads": {
                n: p.grad.detach().cpu().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            },
            "logits": final,
            "loss": losses[-1],
        }
    # Confirm GEMM operands themselves use Float8Tensor, rather than relying
    # solely on config and module flags. Run outside the timing samples.
    linear_impl = importlib.import_module("transformer_engine.pytorch.module.linear")
    original_gemm = linear_impl.general_gemm
    operands = {}

    def measured_gemm(a, b, *args, **kwargs):
        tag = f"{type(a).__name__} x {type(b).__name__}: " + (
            "backward" if kwargs.get("grad") else "forward"
        )
        operands[tag] = operands.get(tag, 0) + 1
        return original_gemm(a, b, *args, **kwargs)

    linear_impl.general_gemm = measured_gemm
    try:
        model.zero_grad(set_to_none=True)
        check = model(tokens, positions, None)
        F.cross_entropy(check.float().flatten(0, 1), labels.flatten()).backward()
    finally:
        linear_impl.general_gemm = original_gemm
    result["gemm_operand_counts"] = operands
    assert any("Float8Tensor" in key for key in operands) == (expected_active > 0)
    for handle in handles:
        handle.remove()
    del model, logits, loss, linears
    gc.collect()
    torch.cuda.empty_cache()
    print(name, "active FP8 linears:", active, "loss:", losses[-1], flush=True)
    return result, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-3b", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ffn-only", action="store_true")
    parser.add_argument("--ffn-projections", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", 0)
    )
    ps.initialize_model_parallel()
    global_vars._GLOBAL_ARGS = SimpleNamespace(transformer_impl="transformer_engine")
    fixture.FULL = args.full_3b
    if args.full_3b:
        fixture.DIMS = dict(
            layers=32,
            hidden=2560,
            heads=40,
            ffn=10240,
            vocab=65536,
            sequence=128,
            batch=1,
            microbatches=1,
        )
        fixture.prepare_minipile()
        source = "/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth"
        sample = torch.load(OUT / "minipile-tokens.pth", weights_only=True).cuda()
        tokens, labels = sample[:, :-1], sample[:, 1:]
    else:
        fixture.prepare_fixture()
        source = OUT / "cropped-native.pth"
        gen = torch.Generator().manual_seed(42)
        tokens, labels = [
            torch.randint(1, 128, (2, 128), generator=gen).cuda() for _ in range(2)
        ]
    native = torch.load(source, map_location="cpu", weights_only=True)
    result = {"runtime": report(), "full_3b": args.full_3b, "cases": {}}
    output = OUT / (
        ("fp8-full-3b" if args.full_3b else "fp8-components")
        + (
            "-ffn-projections"
            if args.ffn_projections
            else "-ffn" if args.ffn_only else ""
        )
        + ".json"
    )

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    row, baseline = run_case(
        "bf16-te", native, tokens, labels, capture=True, repeats=args.repeats
    )
    result["cases"]["bf16-te"] = row
    save()
    cases = [
        ("legacy-disabled", {"recipe": "delayed", "legacy": True}),
        ("delayed", {"recipe": "delayed"}),
        ("delayed-recompute", {"recipe": "delayed", "recompute": True}),
        ("delayed-bf16-wgrad", {"recipe": "delayed", "wgrad": False}),
        ("tensorwise", {"recipe": "tensorwise"}),
        ("tensorwise-first-last-bf16", {"recipe": "tensorwise", "first_last": True}),
    ]
    if args.ffn_only:
        cases = [
            ("delayed-ffn", {"recipe": "delayed", "scope": "ffn"}),
            (
                "delayed-ffn-recompute",
                {"recipe": "delayed", "scope": "ffn", "recompute": True},
            ),
            ("tensorwise-ffn", {"recipe": "tensorwise", "scope": "ffn"}),
            (
                "tensorwise-ffn-recompute",
                {"recipe": "tensorwise", "scope": "ffn", "recompute": True},
            ),
        ]
    if args.ffn_projections:
        cases = [
            (
                "tensorwise-ffn-up-only",
                {"recipe": "tensorwise", "scope": "ffn", "disable_ffn": "down"},
            ),
            (
                "tensorwise-ffn-down-only",
                {"recipe": "tensorwise", "scope": "ffn", "disable_ffn": "up"},
            ),
        ]
    delayed = None
    for name, settings in cases:
        row, snapshot = run_case(
            name,
            native,
            tokens,
            labels,
            baseline=baseline,
            capture=(
                "recompute" in name
                or name in ("delayed", "delayed-ffn", "tensorwise-ffn")
            ),
            repeats=args.repeats,
            **settings,
        )
        if name in ("delayed", "delayed-ffn", "tensorwise-ffn"):
            delayed = snapshot
        elif "recompute" in name:
            row["recompute_loss_exact_vs_plain"] = (
                row["losses"]
                == result["cases"][name.replace("-recompute", "")]["losses"]
            )
            row["recompute_logits_exact_vs_plain"] = torch.equal(
                snapshot["logits"], delayed["logits"]
            )
            row["recompute_gradients_exact_vs_plain"] = all(
                torch.equal(g, delayed["grads"][n])
                for n, g in snapshot["grads"].items()
            )
            assert (
                row["recompute_loss_exact_vs_plain"]
                and row["recompute_logits_exact_vs_plain"]
                and row["recompute_gradients_exact_vs_plain"]
            )
        elif name == "legacy-disabled":
            assert (
                row["gradients"]["relative_l2"] == 0
                and row["logits"]["relative_l2"] == 0
            )
        result["cases"][name] = row
        save()
    result["all_checks_passed"] = True
    save()
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
