"""Compare our actual training runs and decoded TE delayed-scaling checkpoint state."""

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import torch

PATH = Path(__file__).resolve().parent
ROOT = PATH.parents[1]
spec = importlib.util.spec_from_file_location(
    "resume_comparison", PATH / "compare-single-resume.py"
)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def normalized_model(model):
    result = {}
    states = 0
    for key, value in model.items():
        if (
            key.endswith("_extra_state")
            and isinstance(value, torch.Tensor)
            and value.numel()
        ):
            # Only assistant-generated checkpoint paths are opened below.
            state = pickle.loads(value.numpy().tobytes())
            recipe = state["recipe"]
            state["recipe"] = {
                k: v for k, v in vars(recipe).items() if k != "_cached_repr"
            }
            result[key] = state
            states += 1
        else:
            result[key] = value
    return result, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorwise", action="store_true")
    args = parser.parse_args()
    suffix = "-fp8-tensorwise-ffn" if args.tensorwise else "-fp8-ffn"
    resume.SUFFIX = suffix
    saved, restored, continuous = [
        resume.rows(phase) for phase in ("save", "resume", "continuous")
    ]
    assert (
        list(saved) == [1, 2]
        and list(restored) == [3, 4]
        and list(continuous) == [1, 2, 3, 4]
    )
    assert saved | restored == continuous, (saved, restored, continuous)
    assert all(r["skipped"] == r["nan"] == 0 for r in continuous.values())
    captures = {
        phase: json.loads((PATH / f"{suffix[1:]}-{phase}-batches.json").read_text())[
            "events"
        ]
        for phase in ("save", "resume", "continuous")
    }
    hashes = captures["save"][0]["batch_hashes"]
    assert len(hashes) == 4
    assert all(
        event["batch_hashes"] == hashes
        for events in captures.values()
        for event in events
    )
    resumed_losses = [
        e["losses"] for e in captures["resume"] if not e.get("replay_same_weights")
    ]
    continuous_losses = [
        e["losses"] for e in captures["continuous"] if not e.get("replay_same_weights")
    ]
    assert resumed_losses == continuous_losses[2:], (resumed_losses, continuous_losses)
    left = (
        ROOT
        / f"training-smoke/checkpoints/single-resume{suffix}/iter_0000004/mp_rank_00"
    )
    right = (
        ROOT
        / f"training-smoke/checkpoints/single-continuous{suffix}/iter_0000004/mp_rank_00"
    )
    a, b = [
        torch.load(
            p / "model_optim_rng.pt", map_location="cpu", weights_only=False, mmap=True
        )
        for p in (left, right)
    ]
    am, na = normalized_model(a["model"])
    bm, nb = normalized_model(b["model"])
    assert na == nb == (0 if args.tensorwise else 64), (na, nb)
    checks = {"model_including_decoded_fp8_state": resume.compare(am, bm)}
    for key in ("optimizer", "opt_param_scheduler", "rng_state", "iteration"):
        checks[key] = resume.compare(a[key], b[key])
    for path in left.glob("*.pt"):
        if path.name == "model_optim_rng.pt":
            continue
        x, y = [
            torch.load(p / path.name, map_location="cpu", weights_only=False, mmap=True)
            for p in (left, right)
        ]
        checks[path.name] = resume.compare(x, y)
    assert all(not c["differences"] for c in checks.values()), checks
    resume.SUFFIX = "-fixed-eval"
    previous = resume.rows("continuous")
    resume.SUFFIX = "-bf16-fp8-regression"
    # The copied BF16 run uses a distinct log naming convention.
    original_pattern = resume.PATTERN
    rows = {}
    for line in (
        (ROOT / "training-smoke/logs/single-training-bf16-fp8-regression.log")
        .read_text()
        .splitlines()
    ):
        match = original_pattern.search(line)
        if match:
            step, samples, lr, loss, grad, skipped, nan = match.groups()
            rows[int(step)] = dict(
                samples=int(samples),
                lr=float(lr),
                loss=float(loss),
                grad_norm=float(grad),
                skipped=int(skipped),
                nan=int(nan),
            )
    assert rows == previous, (rows, previous)
    result = {
        "training_rows": continuous,
        "checkpoint_comparisons": checks,
        "decoded_te_fp8_state_count": na,
        "model_optimizer_rng_scheduler_and_fp8_scales_exact": True,
        "fixed_eval_samples_across_resume": True,
        "resumed_eval_losses_exact": True,
        "bf16_training_loss_and_gradient_norm_regression_exact": True,
        "te_trusted_pickle_opt_in_used_for_our_own_delayed_scaling_checkpoint": not args.tensorwise,
    }
    (
        PATH
        / (
            "fp8-tensorwise-training-validation.json"
            if args.tensorwise
            else "fp8-training-validation.json"
        )
    ).write_text(json.dumps(result, indent=2) + "\n")
    print("FP8 training/resume and BF16 regression PASS", flush=True)


if __name__ == "__main__":
    main()
