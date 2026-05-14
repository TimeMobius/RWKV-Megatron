from __future__ import annotations

import ast
import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
from einops import einsum, rearrange


REPO_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = REPO_ROOT / "3rdparty" / "Megatron-LM-0.11.0"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    all_to_all_sp2hp,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import divide


SOURCE_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_layer_specs.py"


def _load_rwkv7_bottleneck_class():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(SOURCE_PATH))
    class_node = next(
        node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == "RWKV7Bottleneck"
    )
    class_source = ast.get_source_segment(source, class_node)
    namespace = {
        "MegatronModule": MegatronModule,
        "Tensor": torch.Tensor,
        "TransformerConfig": object,
        "Optional": Optional,
        "Tuple": Tuple,
        "ShardedStateDict": dict,
        "torch": torch,
        "einsum": einsum,
        "rearrange": rearrange,
        "divide": divide,
        "get_tensor_model_parallel_world_size": parallel_state.get_tensor_model_parallel_world_size,
        "set_tensor_model_parallel_attributes": set_tensor_model_parallel_attributes,
        "scatter_to_tensor_model_parallel_region": scatter_to_tensor_model_parallel_region,
        "all_to_all_sp2hp": all_to_all_sp2hp,
        "make_sharded_tensors_for_checkpoint": lambda *args, **kwargs: {},
    }
    exec(class_source, namespace)
    return namespace["RWKV7Bottleneck"]


RWKV7Bottleneck = _load_rwkv7_bottleneck_class()


def _init_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    force_cpu = os.environ.get("RWKV_TEST_FORCE_CPU", "0") == "1"
    use_cuda = torch.cuda.is_available() and not force_cpu
    backend = "nccl" if use_cuda else "gloo"

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    return rank, world_size, device


def _build_identity_bottleneck(device: torch.device):
    config = SimpleNamespace(
        hidden_size=4,
        sequence_parallel=False,
        use_cpu_initialization=False,
        perform_initialization=False,
    )

    patch_ctx = (
        patch("torch.cuda.current_device", lambda: device)
        if device.type == "cpu"
        else contextlib.nullcontext()
    )
    with patch_ctx:
        block = RWKV7Bottleneck(
            config=config,
            bottleneck_size=4,
            to_tensor_parallel=True,
            activation_type="identity",
            use_bias=False,
        ).to(device)

    with torch.no_grad():
        block.weight_down.copy_(torch.eye(4, device=device))
        block.weight_up.copy_(torch.eye(4, device=device))
    return block


def _make_input(device: torch.device):
    return torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0], [100.0, 101.0, 102.0, 103.0]],
            [[1000.0, 1001.0, 1002.0, 1003.0], [1100.0, 1101.0, 1102.0, 1103.0]],
        ],
        device=device,
    )


def _make_local_grad_weights(rank: int, like: torch.Tensor):
    values = torch.arange(like.numel(), dtype=like.dtype, device=like.device).reshape_as(like)
    return values + rank * 100.0


def _assert_tensor_equal(name: str, got: torch.Tensor, expected: torch.Tensor):
    if not torch.equal(got, expected):
        raise AssertionError(f"{name} mismatch\ngot:\n{got}\nexpected:\n{expected}")


def run_non_sequence_parallel_batch_isolation_test():
    rank, world_size, device = _init_distributed()
    try:
        if world_size != 2:
            raise AssertionError(f"expected TP world size 2, got {world_size}")

        block = _build_identity_bottleneck(device)
        x = _make_input(device).requires_grad_(True)

        out = block(x)
        expected_out = torch.chunk(x.detach(), world_size, dim=-1)[rank].contiguous()
        _assert_tensor_equal(f"rank{rank} forward output", out.detach().cpu(), expected_out.cpu())

        local_grad = _make_local_grad_weights(rank, out)
        loss = (out * local_grad).sum()
        loss.backward()

        expected_grad = torch.cat(
            [_make_local_grad_weights(r, expected_out) for r in range(world_size)],
            dim=-1,
        )
        _assert_tensor_equal(f"rank{rank} input grad", x.grad.detach().cpu(), expected_grad.cpu())

        if rank == 0:
            print("PASS: distributed non-SP TP bottleneck keeps batch samples isolated in forward and backward.")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_non_sequence_parallel_batch_isolation_test()
