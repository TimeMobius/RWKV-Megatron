"""Reject precision settings that would silently fail to enable RWKV FP8."""

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rwkv_megatron.arguments import add_rwkv_arguments
from rwkv_megatron.precision import validate_rwkv_precision


@pytest.mark.parametrize("fp8", ["hybrid", "e4m3"])
def test_fp8_requires_te_linears(fp8):
    cfg = SimpleNamespace(fp8=fp8, transformer_impl="transformer_engine")
    validate_rwkv_precision(cfg, have_te=True)
    with pytest.raises(ValueError, match="requires Transformer Engine"):
        validate_rwkv_precision(cfg, have_te=False)
    cfg.transformer_impl = "local"
    with pytest.raises(ValueError, match="requires Transformer Engine"):
        validate_rwkv_precision(cfg, have_te=True)


def test_bf16_fallback_and_wkv_attention_flags():
    validate_rwkv_precision(SimpleNamespace(fp8=None), have_te=False)
    for flag in ("fp8_dot_product_attention", "fp8_multi_head_attention"):
        with pytest.raises(ValueError, match="WKV"):
            validate_rwkv_precision(SimpleNamespace(**{flag: True}), have_te=True)


@pytest.mark.parametrize("deterministic", [False, True])
def test_fla_scan_policy_requires_deterministic_mode(deterministic):
    cfg = SimpleNamespace(fp8=None, deterministic_mode=deterministic)
    with patch("rwkv_megatron.precision.configure_rwkv7_determinism") as configure:
        validate_rwkv_precision(cfg, have_te=False)
        assert configure.call_count == int(deterministic)


def test_ffn_scope_preserves_stored_weight_precision():
    cfg = SimpleNamespace(
        fp8="hybrid", transformer_impl="transformer_engine", fp8_param=True
    )
    with pytest.raises(ValueError, match="BF16 stored weights"):
        validate_rwkv_precision(cfg, have_te=True)
    cfg.fp8_param = False
    cfg.rwkv_fp8_scope = "bad"
    with pytest.raises(ValueError, match="rwkv_fp8_scope"):
        validate_rwkv_precision(cfg, have_te=True)


def test_removed_te_weight_gradient_override_is_explicit():
    cfg = SimpleNamespace(
        fp8="hybrid", transformer_impl="transformer_engine", fp8_wgrad=False
    )
    with patch(
        "rwkv_megatron.precision._has_weight_gradient_override", return_value=False
    ):
        with pytest.warns(RuntimeWarning, match="ignores fp8_wgrad=False"):
            validate_rwkv_precision(cfg, have_te=True)


def test_scope_yaml_and_cli_override(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("rwkv_fp8_scope: all\n")
    monkeypatch.setattr(sys, "argv", ["train", "--rwkv-config", str(path)])
    parser = add_rwkv_arguments(argparse.ArgumentParser())
    assert parser.parse_args().rwkv_fp8_scope == "all"
    assert parser.parse_args(["--rwkv-fp8-scope", "ffn"]).rwkv_fp8_scope == "ffn"
    path.write_text("rwkv_fp8_scope: typo\n")
    with pytest.raises(ValueError, match="rwkv_fp8_scope"):
        add_rwkv_arguments(argparse.ArgumentParser())


@pytest.mark.parametrize("fp8", [True, "true", "fp8", "invalid"])
def test_yaml_rejects_ambiguous_fp8(tmp_path, monkeypatch, fp8):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"language_model": {"fp8": fp8}}))
    monkeypatch.setattr(sys, "argv", ["train", "--rwkv-config", str(path)])
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8", choices=["hybrid", "e4m3"])
    with pytest.raises(ValueError, match="language_model.fp8"):
        add_rwkv_arguments(parser)


@pytest.mark.parametrize("fp8", [False, "hybrid", "e4m3"])
def test_yaml_formats_and_cli_override(tmp_path, monkeypatch, fp8):
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"language_model": {"fp8": fp8}}))
    monkeypatch.setattr(sys, "argv", ["train", "--rwkv-config", str(path)])
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8", choices=["hybrid", "e4m3"])
    add_rwkv_arguments(parser)
    assert parser.parse_args().fp8 == (fp8 or None)
    assert parser.parse_args(["--fp8", "hybrid"]).fp8 == "hybrid"
