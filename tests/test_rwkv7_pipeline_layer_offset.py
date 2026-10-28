"""Uneven PP must load each global RWKV layer exactly once from native weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def loader_offset():
    path = Path(__file__).resolve().parents[1] / "rwkv_megatron/rwkv7_model.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "RWKV7Model")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_layers_start")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_get_layers_start"]


@pytest.mark.parametrize("counts", [(44, 44), (22, 22, 22, 22), (20, 24, 24, 20), (18, 26, 26, 18)])
def test_native_checkpoint_offsets_cover_all_88_layers_without_overlap(counts):
    offset, loaded = 0, []
    for stage, count in enumerate(counts):
        model = SimpleNamespace(
            decoder=SimpleNamespace(layers=[SimpleNamespace(layer_number=n)
                                           for n in range(offset + 1, offset + count + 1)]),
            config=SimpleNamespace(num_layers=88), pre_process=stage == 0,
            post_process=stage == len(counts) - 1)
        start = loader_offset()(model)
        loaded.extend(range(start, start + count))
        assert start == offset
        offset += count
    assert loaded == list(range(88))


def test_virtual_stage_uses_global_numbers_instead_of_physical_stage_rank():
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[SimpleNamespace(layer_number=n) for n in (49, 50)]),
                            config=SimpleNamespace(num_layers=88), pre_process=False, post_process=False)
    assert loader_offset()(model) == 48


def test_noncontiguous_layer_mapping_is_rejected():
    model = SimpleNamespace(decoder=SimpleNamespace(layers=[SimpleNamespace(layer_number=n) for n in (19, 21)]))
    with pytest.raises(ValueError, match="contiguous"):
        loader_offset()(model)
