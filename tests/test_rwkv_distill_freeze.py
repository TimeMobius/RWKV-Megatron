import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rwkv_megatron.model_freeze import (
    FREEZE_COMPONENTS, check_freeze_checkpoint, freeze_policy,
    freeze_rwkv_distill_layers, validate_freeze_settings,
)


class _Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(7, 3)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Embedding()
        self.output_layer = nn.Linear(3, 7, bias=False)
        self.backbone = nn.Linear(3, 3)


def test_freezes_embedding_and_lm_head_but_keeps_backbone_trainable():
    model = _Model()

    counts = freeze_rwkv_distill_layers(model, freeze_embedding=True, freeze_lm_head=True)

    assert counts["embedding"] == counts["lm_head"] == 21
    assert sum(counts.values()) == 42
    assert not model.embedding.word_embeddings.weight.requires_grad
    assert not model.output_layer.weight.requires_grad
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_freezing_defaults_to_no_change():
    model = _Model()

    counts = freeze_rwkv_distill_layers(model)

    assert not any(counts.values())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_freeze_helper_handles_pipeline_stages_without_embedding_or_head():
    model = nn.Linear(3, 3)

    counts = freeze_rwkv_distill_layers(model, freeze_embedding=True, freeze_lm_head=True)

    assert not any(counts.values())
    assert all(parameter.requires_grad for parameter in model.parameters())



def _args(**kwargs):
    return SimpleNamespace(**kwargs)


class _Block(nn.Module):
    def __init__(self, layer_number=1):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(3)
        self.pre_mlp_layernorm = nn.LayerNorm(3)
        self.self_attention = nn.Module()
        self.self_attention.layer_number = layer_number
        for name in ('receptance', 'key', 'value', 'proj'):
            setattr(self.self_attention, f'linear_{name}', nn.Linear(3, 3, bias=False))
        for name in ('learn', 'decay', 'gate', 'v'):
            bottleneck = nn.Module()
            bottleneck.weight_down = nn.Parameter(torch.ones(3, 2))
            bottleneck.weight_up = nn.Parameter(torch.ones(2, 3))
            if name != 'gate':
                bottleneck.bias = nn.Parameter(torch.ones(3))
            setattr(self.self_attention, f'bottleneck_{name}', bottleneck)
        self.self_attention.mix_coeff_input = nn.Parameter(torch.ones(6, 3))
        self.self_attention.group_norm = nn.LayerNorm(3)
        self.mlp = nn.Module()
        self.mlp.linear_up = nn.Linear(3, 6, bias=False)
        self.mlp.linear_down = nn.Linear(6, 3, bias=False)
        self.mlp.mix_coeffs = nn.Parameter(torch.ones(3))


class _FullModel(_Model):
    def __init__(self, layer_number=1):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_Block(layer_number)])
        self.decoder.final_layernorm = nn.LayerNorm(3)


@pytest.mark.parametrize('component,fragment', [
    ('embedding', 'embedding.word_embeddings.'), ('lm_head', 'output_layer.'),
    ('attn_norm', '.input_layernorm.'), ('ffn_norm', '.pre_mlp_layernorm.'),
    ('final_norm', 'decoder.final_layernorm.'), ('ffn_key', '.mlp.linear_up.'),
    ('ffn_value', '.mlp.linear_down.'), ('attn_receptance', '.self_attention.linear_receptance.'),
    ('attn_key', '.self_attention.linear_key.'), ('attn_value', '.self_attention.linear_value.'),
    ('attn_output', '.self_attention.linear_proj.'), ('lora', '.self_attention.bottleneck_'),
    ('token_shift', '.self_attention.mix_coeff_input'), ('ffn_receptance', 'ABSENT'),
])
def test_each_component_freezes_only_its_own_parameters(component, fragment):
    model = _FullModel()
    counts = freeze_rwkv_distill_layers(model, args=_args(**{f'rwkv_freeze_{component}': True}))
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (fragment not in name), name
    assert counts[component] == sum(p.numel() for n, p in model.named_parameters() if fragment in n)


def test_native_patterns_use_global_pipeline_layer_numbers_and_train_wins():
    model = _FullModel(layer_number=7)
    freeze_rwkv_distill_layers(model, args=_args(
        rwkv_freeze_attn_key=True,
        rwkv_freeze_extra_patterns=['blocks.6.*'],
        rwkv_train_extra_patterns=['blocks.6.att.key.weight']))
    for name, p in model.named_parameters():
        assert p.requires_grad == (not name.startswith('decoder.layers.') or '.linear_key.' in name)


def test_megatron_patterns_are_global_and_cannot_accidentally_match_local_layer_zero():
    model = _FullModel(layer_number=7)
    freeze_rwkv_distill_layers(model, args=_args(rwkv_freeze_extra_patterns=['decoder.layers.0.*']))
    assert all(p.requires_grad for p in model.parameters())
    freeze_rwkv_distill_layers(model, args=_args(rwkv_freeze_extra_patterns=['decoder.layers.6.*']))
    assert all(not p.requires_grad for p in model.decoder.layers[0].parameters())


def test_reject_partial_fused_shift_freeze():
    with pytest.raises(ValueError, match='Partial attention token-shift'):
        freeze_rwkv_distill_layers(_FullModel(), args=_args(rwkv_freeze_extra_patterns=['*.att.x_r']))
    model = _FullModel()
    freeze_rwkv_distill_layers(model, args=_args(rwkv_freeze_extra_patterns=['*.att.x_*']))
    assert not model.decoder.layers[0].self_attention.mix_coeff_input.requires_grad


def test_frozen_head_still_propagates_input_gradient_and_adam_never_updates_frozen_weights():
    torch.manual_seed(123)
    model = _Model()
    reference = _Model()
    reference.load_state_dict(model.state_dict())
    freeze_rwkv_distill_layers(model, True, True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    ids = torch.tensor([1, 2, 3])
    def loss(m):
        return m.output_layer(m.backbone(m.embedding.word_embeddings(ids))).square().mean()
    assert torch.equal(loss(model), loss(reference))
    loss(model).backward()
    loss(reference).backward()
    torch.testing.assert_close(model.backbone.weight.grad, reference.backbone.weight.grad, rtol=0, atol=0)
    assert model.output_layer.weight.grad is None
    assert model.embedding.word_embeddings.weight.grad is None
    optimizer.step()
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]) == (not p.requires_grad)


def test_tied_embedding_head_conflicts_are_rejected():
    model = _Model()
    model.share_embeddings_and_output_weights = True
    model.output_layer.weight = model.embedding.word_embeddings.weight
    with pytest.raises(ValueError, match='Tied embedding/head'):
        freeze_rwkv_distill_layers(model, True, False)
    freeze_rwkv_distill_layers(model, True, True)
    assert not model.embedding.word_embeddings.weight.requires_grad


@pytest.mark.parametrize('value', ['true', 1, None])
def test_validate_rejects_non_boolean_switches(value):
    with pytest.raises(ValueError, match='boolean'):
        validate_freeze_settings({'rwkv_freeze_lora': value})


@pytest.mark.parametrize('value', ['*.weight', [1], [''], None])
def test_validate_rejects_invalid_patterns(value):
    with pytest.raises(ValueError, match='patterns'):
        validate_freeze_settings({'rwkv_train_extra_patterns': value})


@pytest.mark.parametrize('component', FREEZE_COMPONENTS)
def test_strategy_change_requires_explicit_optimizer_reset_in_either_direction(component):
    plain = _args()
    frozen = _args(**{f'rwkv_freeze_{component}': True})
    for old, new in ((plain, frozen), (frozen, plain)):
        with pytest.raises(ValueError, match='no_load_optim'):
            check_freeze_checkpoint(new, old)
        new.no_load_optim = True
        check_freeze_checkpoint(new, old)
        del new.no_load_optim


def test_same_policy_legacy_defaults_and_pattern_order_resume_without_reset():
    check_freeze_checkpoint(_args(), _args())
    a = _args(rwkv_freeze_embedding=True, rwkv_freeze_extra_patterns=['a', 'b', 'a'])
    b = _args(rwkv_freeze_embedding=True, rwkv_freeze_extra_patterns=['b', 'a'])
    check_freeze_checkpoint(a, b)
    assert freeze_policy(a) == freeze_policy(b)
    with pytest.raises(ValueError, match='no_load_optim'):
        check_freeze_checkpoint(_args(rwkv_train_extra_patterns=['*key*']), _args())


def test_missing_metadata_is_safe_only_without_a_new_freeze_policy():
    check_freeze_checkpoint(_args(), None)
    with pytest.raises(ValueError, match='no freeze metadata'):
        check_freeze_checkpoint(_args(rwkv_freeze_lora=True), None)
    check_freeze_checkpoint(_args(rwkv_freeze_lora=True), None, release=True)
    check_freeze_checkpoint(_args(rwkv_freeze_lora=True, finetune=True), _args())


def test_all_switches_and_patterns_are_exposed_by_argument_parser():
    from rwkv_megatron.arguments import add_rwkv_arguments
    args = add_rwkv_arguments(argparse.ArgumentParser()).parse_args([
        '--rwkv-freeze-lora', '--rwkv-freeze-attn-key',
        '--rwkv-freeze-extra-patterns', 'blocks.0.*', '--rwkv-train-extra-patterns', '*.a1'])
    assert args.rwkv_freeze_lora and args.rwkv_freeze_attn_key
    assert args.rwkv_freeze_extra_patterns == ['blocks.0.*']
    assert args.rwkv_train_extra_patterns == ['*.a1']


def test_checkpoint_guard_intercepts_before_optimizer_load_and_restores_adapter(monkeypatch):
    from megatron.training import checkpointing
    from rwkv_megatron.model_freeze import freeze_checkpoint_guard
    state = {'args': _args()}
    def load(*a, **kw):
        return state, 'checkpoint', False, None
    monkeypatch.setattr(checkpointing, '_load_base_checkpoint', load)
    with pytest.raises(ValueError, match='no_load_optim'):
        with freeze_checkpoint_guard():
            checkpointing._load_base_checkpoint('/unused', _args(rwkv_freeze_lora=True), rank0=True)
    assert checkpointing._load_base_checkpoint is load
    with freeze_checkpoint_guard():
        assert checkpointing._load_base_checkpoint('/unused', _args())[0] is state
    assert checkpointing._load_base_checkpoint is load


def test_yaml_freeze_settings_validate_and_populate_cli_defaults(tmp_path, monkeypatch):
    import sys
    import yaml
    from rwkv_megatron.arguments import add_rwkv_arguments
    path = tmp_path / 'freeze.yaml'
    config = {'rwkv_freeze_attn_norm': True,
              'rwkv_freeze_extra_patterns': ['blocks.3.*'],
              'rwkv_train_extra_patterns': ['blocks.3.att.w1']}
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, 'argv', ['program', '--rwkv-config', str(path)])
    parsed = add_rwkv_arguments(argparse.ArgumentParser()).parse_args()
    for name, value in config.items():
        assert getattr(parsed, name) == value
    config['rwkv_freeze_attn_norm'] = 'true'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='boolean'):
        add_rwkv_arguments(argparse.ArgumentParser())


@pytest.mark.parametrize('mode', ['no_load_optim', 'finetune'])
def test_strategy_change_cannot_reuse_old_positional_master_weights_even_when_resetting(mode):
    with pytest.raises(ValueError, match='disable load_main_params_from_ckpt'):
        check_freeze_checkpoint(_args(rwkv_freeze_lora=True, load_main_params_from_ckpt=True,
                                     **{mode: True}), _args())


def test_native_partial_shift_train_override_cannot_unfreeze_all_of_a_globally_frozen_tensor():
    with pytest.raises(ValueError, match='Partial attention token-shift'):
        freeze_rwkv_distill_layers(_FullModel(), args=_args(
            rwkv_freeze_extra_patterns=['decoder.layers.*.self_attention.mix_coeff_input'],
            rwkv_train_extra_patterns=['*.att.x_r']))
