"""
Regression test for RWKV7 rwkv-lm debug reference guarding.

Expected workflow:
1. Before fixing rwkv7_model.py, enabling check_against_pretrained on the rwkv-lm
   loader would silently stash a raw state_dict and only fail later in forward().
2. After the fix, the loader should reject that unsupported path immediately with a
   clear error.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "rwkv_megatron" / "rwkv7_model.py"


def _load_from_rwkv_lm():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(SOURCE_PATH))

    model_node = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.ClassDef) and node.name == "RWKV7Model"
    )
    method_node = next(
        node
        for node in model_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "from_rwkv_lm"
    )

    selected_module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(selected_module)
    code = compile(selected_module, str(SOURCE_PATH), "exec")

    namespace = {
        "torch": SimpleNamespace(no_grad=lambda fn: fn),
    }
    exec(code, namespace)
    return namespace["from_rwkv_lm"]


def test_rwkv_lm_check_against_pretrained_raises_clear_error():
    from_rwkv_lm = _load_from_rwkv_lm()

    try:
        from_rwkv_lm(object(), {}, True)
    except NotImplementedError as exc:
        message = str(exc)
        assert "rwkv-huggingface" in message
        assert "rwkv-lm" in message
        assert "callable reference model" in message
    else:
        raise AssertionError(
            "from_rwkv_lm should reject check_against_pretrained=True with a clear error"
        )


if __name__ == "__main__":
    test_rwkv_lm_check_against_pretrained_raises_clear_error()
    print("PASS: RWKV7 rwkv-lm debug reference path fails fast with a clear error.")
