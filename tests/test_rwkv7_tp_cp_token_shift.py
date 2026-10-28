"""Distributed token-shift values and gradients follow logical CP-major TP order."""

import ast
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rwkv_megatron.token_shift import tp_cp_shift_neighbors

ROOT = Path(__file__).resolve().parents[1]


def load_shift(parallel_state):
    # Load the real autograd functions without initializing the full model stack.
    source = ROOT / "rwkv_megatron/rwkv7_layer_specs.py"
    names = {
        "RWKV7TokenShiftSequenceParallel",
        "RWKV7TokenShiftContextParallel",
        "_token_shift",
    }
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if getattr(node, "name", None) in names
    ]
    namespace = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "dist": dist,
        "custom_fwd": lambda fn: fn,
        "custom_bwd": lambda fn: fn,
        "tp_cp_shift_neighbors": tp_cp_shift_neighbors,
    }
    for name in (
        "get_tensor_model_parallel_group",
        "get_tensor_model_parallel_world_size",
        "get_tensor_model_parallel_rank",
        "get_context_parallel_group",
        "get_context_parallel_world_size",
        "get_context_parallel_rank",
        "get_tensor_and_context_parallel_group",
    ):
        namespace[name] = getattr(parallel_state, name)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace["_token_shift"]


def run_worker(rank, tp, cp, order, store):
    from megatron.core import parallel_state

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=store,
        rank=rank,
        world_size=tp * cp,
        timeout=timedelta(seconds=120),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp, context_parallel_size=cp, order=order
    )
    shift = load_shift(parallel_state)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    failures = []
    for sp in (False, True):
        for length in (1, 3):
            total = length * tp * cp
            original = torch.arange(
                total * 3 * 7, device="cuda", dtype=torch.float64
            ).reshape(total, 3, 7)
            expected = torch.cat((torch.zeros_like(original[:1]), original[:-1]))
            derivative = (original + 1).square()
            expected_grad = torch.cat(
                (derivative[1:], torch.zeros_like(derivative[:1]))
            )
            width = length if sp else length * tp
            index = cp_rank * tp + tp_rank if sp else cp_rank
            section = slice(index * width, (index + 1) * width)
            for dim in (0, 1):
                local = original[section]
                incoming = derivative[section]
                target = expected[section]
                target_grad = expected_grad[section]
                if dim == 1:
                    local, incoming, target, target_grad = [
                        x.transpose(0, 1)
                        for x in (local, incoming, target, target_grad)
                    ]
                local = local.detach().clone().requires_grad_()
                result = shift(local, dim, sp)
                (result * incoming).sum().backward()
                if not torch.equal(result, target) or not torch.equal(
                    local.grad, target_grad
                ):
                    failures.append((sp, length, dim))
    bad = torch.tensor([len(failures)], device="cuda", dtype=torch.int32)
    dist.all_reduce(bad)
    count = bad.item()
    dist.barrier()
    parallel_state.destroy_model_parallel()
    tp_cp_shift_neighbors.cache_clear()
    dist.destroy_process_group()
    assert count == 0, (rank, failures, count)


@pytest.mark.parametrize(
    ("tp", "cp", "order"),
    [
        pytest.param(2, 1, "tp-cp-ep-dp-pp", id="TP2-CP1"),
        pytest.param(1, 2, "tp-cp-ep-dp-pp", id="TP1-CP2"),
        pytest.param(2, 2, "tp-cp-ep-dp-pp", id="TP2-CP2"),
        pytest.param(2, 2, "cp-tp-ep-dp-pp", id="CP2-before-TP2"),
        pytest.param(4, 2, "tp-cp-ep-dp-pp", id="TP4-CP2"),
        pytest.param(1, 4, "tp-cp-ep-dp-pp", id="TP1-CP4"),
        pytest.param(2, 4, "tp-cp-ep-dp-pp", id="TP2-CP4"),
        pytest.param(2, 4, "cp-tp-ep-dp-pp", id="CP4-before-TP2"),
        pytest.param(1, 8, "tp-cp-ep-dp-pp", id="TP1-CP8"),
    ],
)
def test_tp_cp_shift_matches_global_values_and_gradients(tp, cp, order, tmp_path):
    if torch.cuda.device_count() < tp * cp:
        pytest.skip(f"Requires {tp * cp} GPUs")
    mp.spawn(
        run_worker,
        args=(tp, cp, order, f'file://{tmp_path / "store"}'),
        nprocs=tp * cp,
        join=True,
    )
