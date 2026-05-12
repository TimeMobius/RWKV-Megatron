from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Tuple
from unittest.mock import patch

import torch
from einops import einsum, rearrange


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_layer_specs.py"
TP_SIZE = 2
HIDDEN_SIZE = 4


class _MinimalMegatronModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


def _unpatched_comm(*args, **kwargs):
    raise RuntimeError("communication helper must be patched by the test")


def _load_rwkv7_bottleneck_class():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(SOURCE_PATH))
    class_node = next(
        node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == "RWKV7Bottleneck"
    )
    class_source = ast.get_source_segment(source, class_node)
    namespace = {
        "MegatronModule": _MinimalMegatronModule,
        "Tensor": torch.Tensor,
        "TransformerConfig": object,
        "Optional": __import__("typing").Optional,
        "Tuple": Tuple,
        "ShardedStateDict": dict,
        "torch": torch,
        "einsum": einsum,
        "rearrange": rearrange,
        "divide": lambda x, y: x // y,
        "get_tensor_model_parallel_world_size": lambda: TP_SIZE,
        "set_tensor_model_parallel_attributes": lambda *args, **kwargs: None,
        "scatter_to_tensor_model_parallel_region": _unpatched_comm,
        "all_to_all_sp2hp": _unpatched_comm,
        "make_sharded_tensors_for_checkpoint": lambda *args, **kwargs: {},
    }
    exec(class_source, namespace)
    return namespace["RWKV7Bottleneck"]


RWKV7Bottleneck = _load_rwkv7_bottleneck_class()


def _build_identity_bottleneck(sequence_parallel: bool):
    config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        sequence_parallel=sequence_parallel,
        use_cpu_initialization=False,
        perform_initialization=False,
    )
    with patch("torch.cuda.current_device", lambda: "cpu"):
        block = RWKV7Bottleneck(
            config=config,
            bottleneck_size=HIDDEN_SIZE,
            to_tensor_parallel=True,
            activation_type="identity",
            use_bias=False,
        )
    with torch.no_grad():
        block.weight_down.copy_(torch.eye(HIDDEN_SIZE))
        block.weight_up.copy_(torch.eye(HIDDEN_SIZE))
    return block


def _run_with_patched_comm(
    block: torch.nn.Module,
    input_tensor: torch.Tensor,
    comm_name: str,
    patched_comm: Callable[[torch.Tensor], torch.Tensor],
):
    forward_globals = block.forward.__func__.__globals__
    original_comm = forward_globals[comm_name]
    forward_globals[comm_name] = patched_comm
    try:
        return block(input_tensor)
    finally:
        forward_globals[comm_name] = original_comm


def run_non_sequence_parallel_outputs(x: torch.Tensor):
    outputs = []
    for rank in range(TP_SIZE):
        block = _build_identity_bottleneck(sequence_parallel=False)

        def fake_scatter(inp: torch.Tensor, rank_: int = rank):
            return torch.chunk(inp, TP_SIZE, dim=-1)[rank_].contiguous()

        outputs.append(
            _run_with_patched_comm(
                block=block,
                input_tensor=x.clone(),
                comm_name="scatter_to_tensor_model_parallel_region",
                patched_comm=fake_scatter,
            ).detach()
        )
    return outputs


def _emulate_all_to_all_sp2hp(per_rank_inputs, dst_rank: int):
    concat_tensors = []
    for input_tensor in per_rank_inputs:
        flattened = input_tensor.reshape(-1, input_tensor.shape[-1])
        split_hidden = torch.split(flattened, flattened.shape[-1] // TP_SIZE, dim=1)
        concat_tensors.append(torch.cat(split_hidden, dim=0))

    split_along_sequence = [torch.chunk(tensor, TP_SIZE, dim=0) for tensor in concat_tensors]
    return torch.cat([split_along_sequence[src_rank][dst_rank] for src_rank in range(TP_SIZE)], dim=0)


def run_sequence_parallel_outputs(global_x: torch.Tensor):
    local_inputs = list(torch.chunk(global_x, TP_SIZE, dim=0))
    per_rank_flat_inputs = [rearrange(local_x, "t b d -> t (b d)") for local_x in local_inputs]
    outputs = []

    for rank, local_x in enumerate(local_inputs):
        block = _build_identity_bottleneck(sequence_parallel=True)

        def fake_all_to_all(inp: torch.Tensor, rank_: int = rank):
            expected_local_input = per_rank_flat_inputs[rank_]
            assert torch.equal(inp, expected_local_input), (
                f"local rank {rank_} forward input changed unexpectedly\n"
                f"got:\n{inp}\nexpected:\n{expected_local_input}"
            )
            return _emulate_all_to_all_sp2hp(per_rank_flat_inputs, dst_rank=rank_)

        outputs.append(
            _run_with_patched_comm(
                block=block,
                input_tensor=local_x.clone(),
                comm_name="all_to_all_sp2hp",
                patched_comm=fake_all_to_all,
            ).detach()
        )
    return outputs


def expected_hidden_shards(global_x: torch.Tensor):
    return [chunk.contiguous() for chunk in torch.chunk(global_x, TP_SIZE, dim=-1)]


def assert_tensor_equal(name: str, got: torch.Tensor, expected: torch.Tensor):
    assert torch.equal(got, expected), f"{name} mismatch\ngot:\n{got}\nexpected:\n{expected}"
