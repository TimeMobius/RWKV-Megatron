"""
Static contract audit for RWKV7 regression fixes and TP/PP/SP protocol wiring.

This file intentionally avoids importing torch or Megatron so it can run in
lightweight environments. It uses Python AST inspection to verify that the key
code paths still implement the contracts established by the previous task fixes.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYER_SPECS_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_layer_specs.py"
MODEL_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_model.py"
SCHEDULES_PATH = (
    REPO_ROOT
    / "3rdparty"
    / "Megatron-LM-0.11.0"
    / "megatron"
    / "core"
    / "pipeline_parallel"
    / "schedules.py"
)
TRANSFORMER_LAYER_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_transformer_layer.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _get_class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _get_function(container, name: str) -> ast.FunctionDef:
    return next(node for node in container.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _node_dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _string_constants(node: ast.AST) -> list[str]:
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.append(child.value)
    return values


def test_task1_bottleneck_sp_path_flattens_tokens_without_batch_mixing():
    source = LAYER_SPECS_PATH.read_text(encoding="utf-8")

    assert 'rearrange(expanded, "t b d -> (t b) d")' in source, (
        "SP bottleneck path should flatten as (t b) d"
    )
    assert 'b=x.shape[1]' in source, (
        "SP bottleneck path should reshape back with b=x.shape[1] to avoid sample mixing"
    )
    assert "all_to_all_sp2hp" in source, "SP bottleneck path should use all_to_all_sp2hp"


def test_task2_and_task3_scratch_init_helpers_are_wired_in():
    module = _parse(LAYER_SPECS_PATH)
    helper_names = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_rwkv7_")
    }
    assert "_rwkv7_time_mix_init_tensors" in helper_names
    assert "_rwkv7_channel_mix_init_tensor" in helper_names

    timemix_cls = _get_class(module, "RWKV7TimeMix")
    timemix_init = _get_function(timemix_cls, "__init__")
    timemix_dump = _node_dump(timemix_init)
    assert "_rwkv7_time_mix_init_tensors" in timemix_dump, (
        "RWKV7TimeMix.__init__ should call the official scratch-init helper"
    )

    channelmix_cls = _get_class(module, "RWKV7ChannelMix")
    channelmix_set_layer_number = _get_function(channelmix_cls, "set_layer_number")
    channelmix_dump = _node_dump(channelmix_set_layer_number)
    assert "_rwkv7_channel_mix_init_tensor" in channelmix_dump, (
        "RWKV7ChannelMix.set_layer_number should call the official scratch-init helper"
    )


def test_task5_groupnorm_eps_contract_matches_official_formula():
    module = _parse(LAYER_SPECS_PATH)

    groupnorm_cls = _get_class(module, "GroupNorm")
    groupnorm_init = _get_function(groupnorm_cls, "__init__")
    assignments = [
        node for node in ast.walk(groupnorm_init) if isinstance(node, ast.Assign)
    ]
    eps_assign = next(
        node
        for node in assignments
        if any(isinstance(target, ast.Attribute) and target.attr == "eps" for target in node.targets)
    )
    assert isinstance(eps_assign.value, ast.Name) and eps_assign.value.id == "eps", (
        "GroupNorm should respect the explicit eps argument"
    )

    timemix_cls = _get_class(module, "RWKV7TimeMix")
    timemix_init = _get_function(timemix_cls, "__init__")
    timemix_source = _node_dump(timemix_init)
    assert "group_norm_eps" in timemix_source
    assert "layernorm_epsilon" in timemix_source
    assert "num_attention_heads" in timemix_source
    assert "hidden_size" in timemix_source


def test_task4_hf_lora_loader_mapping_uses_correct_weight_orientation():
    module = _parse(MODEL_PATH)
    model_cls = _get_class(module, "RWKV7Model")
    from_hf = _get_function(model_cls, "from_mobius_huggingface")
    from_hf_dump = _node_dump(from_hf)

    assert ".lora.0.weight" in from_hf_dump
    assert ".lora.2.weight" in from_hf_dump
    assert "weight_down" in from_hf_dump
    assert "weight_up" in from_hf_dump
    assert "transpose" in from_hf_dump, (
        "HF LoRA loader should transpose HuggingFace linear weights into RWKV7Bottleneck layout"
    )


def test_task6_rwkv_lm_check_against_pretrained_fails_fast():
    module = _parse(MODEL_PATH)
    model_cls = _get_class(module, "RWKV7Model")
    from_rwkv_lm = _get_function(model_cls, "from_rwkv_lm")
    raises = [node for node in ast.walk(from_rwkv_lm) if isinstance(node, ast.Raise)]
    not_impl = next(
        node for node in raises if isinstance(node.exc, ast.Call) and _call_name(node.exc) == "NotImplementedError"
    )
    messages = " ".join(_string_constants(not_impl))
    assert "rwkv-huggingface" in messages
    assert "rwkv-lm" in messages
    assert "callable reference model" in messages


def test_task7_tp_pp_sp_protocol_stays_consistent_across_three_codepaths():
    schedules = _parse(SCHEDULES_PATH)
    transformer_layer = _parse(TRANSFORMER_LAYER_PATH)
    layer_specs = _parse(LAYER_SPECS_PATH)

    get_tensor_shapes = _get_function(schedules, "get_tensor_shapes")
    schedules_dump = _node_dump(get_tensor_shapes)
    assert "rwkv7_value_residual" in schedules_dump
    assert "tp_group" in schedules_dump
    assert "value=2" in schedules_dump, "SP+PP schedule should reserve hidden states and local-head v0"

    transformer_cls = _get_class(transformer_layer, "RWKV7TransformerLayer")
    transformer_forward = _get_function(transformer_cls, "forward")
    transformer_source = TRANSFORMER_LAYER_PATH.read_text(encoding="utf-8")
    assert "values = hidden_states[1:]" in transformer_source
    assert "torch.cat((normalized.unsqueeze(0), values), dim=0)" in transformer_source
    assert "values = attention_output[1:]" in transformer_source

    timemix_cls = _get_class(layer_specs, "RWKV7TimeMix")
    timemix_forward = _get_function(timemix_cls, "forward")
    timemix_source = LAYER_SPECS_PATH.read_text(encoding="utf-8")
    assert "unpack_sharded_value(packed_v0, tp_size)" in timemix_source
    assert "gather_from_tensor_model_parallel_region(value)" in timemix_source
    assert "v0[:, :, start_idx:end_idx]" in timemix_source
    assert "torch.lerp(value, v0_shard, vt)" in timemix_source
    assert "pack_sharded_value(v0, tp_size)" in timemix_source
    assert "torch.stack((output, packed_v0), dim=0)" in timemix_source


def test_context_parallel_is_still_explicitly_rejected_for_rwkv_token_shift():
    module = _parse(LAYER_SPECS_PATH)
    cls = _get_class(module, "RWKV7TokenShiftSequenceParallel")
    forward = _get_function(cls, "forward")
    source = ast.unparse(forward)
    assert "get_context_parallel_world_size() == 1" in source
    assert "Context parallel for RWKV not yet implemented" in source


if __name__ == "__main__":
    test_task1_bottleneck_sp_path_flattens_tokens_without_batch_mixing()
    test_task2_and_task3_scratch_init_helpers_are_wired_in()
    test_task5_groupnorm_eps_contract_matches_official_formula()
    test_task4_hf_lora_loader_mapping_uses_correct_weight_orientation()
    test_task6_rwkv_lm_check_against_pretrained_fails_fast()
    test_task7_tp_pp_sp_protocol_stays_consistent_across_three_codepaths()
    test_context_parallel_is_still_explicitly_rejected_for_rwkv_token_shift()
    print("PASS: RWKV7 static contract audit matches task 1-7 fixes and TP/PP/SP protocol.")
