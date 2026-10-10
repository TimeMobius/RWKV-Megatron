"""All default RWKV7 norm sites share the TE/Apex capability decision."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from rwkv_megatron import normalization
from rwkv_megatron.rwkv7_layer_specs import (
    get_rwkv7_layer_spec,
    get_rwkv7_with_transformer_engine_spec,
)
from rwkv_megatron.rwkv7_model import RWKV7Model


@pytest.mark.parametrize('available,new_enough', [(True, True), (True, False), (False, False)])
def test_both_layer_specs_select_the_same_capable_backend(available, new_enough):
    with patch.object(normalization, 'HAVE_TE', available), patch.object(
        normalization, 'is_te_min_version', return_value=new_enough
    ) as version:
        if available and new_enough:
            first, second = get_rwkv7_layer_spec(), get_rwkv7_with_transformer_engine_spec()
            expected = normalization.TENorm
            version.assert_called_with('1.9.0')
        else:
            with pytest.warns(RuntimeWarning, match='using Apex'):
                first, second = get_rwkv7_layer_spec(), get_rwkv7_with_transformer_engine_spec()
            expected = FusedLayerNorm
            if not available:
                version.assert_not_called()
        for spec in (first, second):
            assert spec.submodules.input_layernorm is expected
            assert spec.submodules.pre_mlp_layernorm is expected


@pytest.mark.parametrize('positional', [False, True])
@pytest.mark.parametrize('pre_process', [False, True])
@pytest.mark.parametrize('fallback', [False, True])
def test_model_uses_same_ln0_and_explicit_final_norm_without_mutating_spec(positional, pre_process, fallback):
    class Norm(torch.nn.Module):
        def __init__(self, config, hidden_size, eps):
            super().__init__()
            assert hidden_size == 8 and eps == 1e-5

    config = SimpleNamespace(hidden_size=8, layernorm_epsilon=1e-5)
    original = TransformerBlockSubmodules(layer_specs=[ModuleSpec(module=object)], layer_norm=object)
    group = object()
    groups = SimpleNamespace(pp=group)

    def parent(model, config, transformer_layer_spec, **kwargs):
        torch.nn.Module.__init__(model)
        model.pre_process = pre_process
        assert transformer_layer_spec.layer_norm is Norm
        assert transformer_layer_spec.layer_specs == original.layer_specs
        assert original.layer_norm is object

    def get_spec(config, spec, vp_stage, pp_rank):
        assert spec is original and vp_stage == 1 and pp_rank == 2
        return spec

    with patch.object(normalization, 'HAVE_TE', not fallback), patch.object(
        normalization, 'is_te_min_version', return_value=True
    ), patch.object(normalization, 'TENorm', Norm), patch.object(
        normalization, 'FusedLayerNorm', Norm
    ), patch.object(GPTModel, '__init__', parent), patch(
        'rwkv_megatron.rwkv7_model._get_block_submodules', side_effect=get_spec
    ), patch('rwkv_megatron.rwkv7_model.get_pg_rank', side_effect=lambda g: 2 if g is group else None):
        kwargs = dict(pg_collection=groups, vp_stage=1, pre_process=pre_process)
        def construct():
            if positional:
                return RWKV7Model(config, original, **kwargs)
            return RWKV7Model(config, transformer_layer_spec=original, **kwargs)
        if fallback:
            with pytest.warns(RuntimeWarning, match='using Apex'):
                model = construct()
        else:
            model = construct()
        assert isinstance(model.pre_norm, Norm) if pre_process else not hasattr(model, 'pre_norm')
