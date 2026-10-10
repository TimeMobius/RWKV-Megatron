"""L2Wrap gradient, mask/scaling, CE backend and YAML regression tests."""
import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from rwkv_megatron.arguments import add_rwkv_arguments
from rwkv_megatron.l2wrap import wrap_token_loss
from rwkv_megatron.rwkv7_model import RWKV7Model


def ce(logits, labels):
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').view_as(labels).T.contiguous()


@pytest.mark.parametrize('scale', [0., 1., 8., .25])
@pytest.mark.parametrize('masked', [False, True, 'all'])
def test_exact_max_logit_gradient_and_mask_scaling(scale, masked):
    torch.manual_seed(19)
    logits = torch.randn(4, 2, 11, requires_grad=True)
    # Include a tie, maxima on either end, and negative maxima.
    with torch.no_grad():
        logits[0, 0, 0] = logits[0, 0, 10] = 8
        logits[1, 0] -= logits[1, 0].max() + 2
    labels = torch.randint(0, 11, (4, 2))
    mask = torch.ones(2, 4)
    if masked is True: mask[0, 1] = mask[1, 2] = 0
    if masked == 'all': mask.zero_()
    base = ce(logits, labels)
    wrapped = wrap_token_loss(base, logits, 1e-4)
    assert torch.equal(base, wrapped)
    upstream = scale * mask / mask.sum().clamp_min(1)
    actual = torch.autograd.grad(wrapped, logits, upstream, retain_graph=True)[0]
    ce_grad = torch.autograd.grad(base, logits, upstream)[0]
    maxima, indices = logits.detach().max(-1)
    expected_extra = torch.zeros_like(logits).scatter_(-1, indices.unsqueeze(-1),
        (1e-4 * maxima * upstream.T).unsqueeze(-1))
    torch.testing.assert_close(actual, ce_grad + expected_extra, rtol=0, atol=0)
    assert torch.equal(actual.transpose(0, 1)[mask == 0], torch.zeros_like(actual.transpose(0, 1)[mask == 0]))


def test_matches_fla_scalar_wrapper_without_mask():
    from fla.modules.l2warp import l2_warp
    torch.manual_seed(8)
    logits = torch.randn(3, 2, 17, requires_grad=True)
    labels = torch.randint(0, 17, (3, 2))
    base = ce(logits, labels)
    a = wrap_token_loss(base, logits, 1e-4).mean() * 13
    b = l2_warp(base.mean(), logits.transpose(0, 1), 1e-4) * 13
    first = torch.autograd.grad(a, logits, retain_graph=True)[0]
    second = torch.autograd.grad(b, logits)[0]
    assert torch.equal(a, b)
    torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('per_token', [False, True])
def test_microbatch_accumulation_and_token_count_division(per_token):
    torch.manual_seed(12)
    whole = torch.randn(3, 4, 13, requires_grad=True)
    labels = torch.randint(0, 13, (3, 4))
    mask = torch.ones(4, 3)
    mask[[0, 2], 0] = 0  # Equal valid counts in the two microbatches.
    full_loss = wrap_token_loss(ce(whole, labels), whole, 1e-4)
    expected = torch.autograd.grad((full_loss * mask).sum()/mask.sum(), whole)[0]
    pieces = []
    for start in (0, 2):
        part = whole[:, start:start+2].detach().clone().requires_grad_()
        local_mask = mask[start:start+2]
        loss = wrap_token_loss(ce(part, labels[:, start:start+2]), part, 1e-4)
        scalar = (loss * local_mask).sum()
        if not per_token: scalar = scalar/local_mask.sum()/2
        grad = torch.autograd.grad(scalar, part)[0]
        if per_token: grad = grad/mask.sum()  # Megatron's final gradient normalization.
        pieces.append(grad)
    torch.testing.assert_close(torch.cat(pieces, 1), expected, rtol=1e-6, atol=1e-8)


def minimal_model(coefficient):
    model = RWKV7Model.__new__(RWKV7Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.vocab_size = 11
    model.tp_group = None
    model.l2wrap_coefficient = coefficient
    return model


def test_model_captures_max_before_mutating_ce_backend():
    model = minimal_model(1e-4)
    original = torch.randn(3, 2, 11)
    logits = original.clone().requires_grad_()
    labels = torch.randint(0, 11, (2, 3))
    def mutating_backend(x, targets, group):
        loss = ce(x.clone(), targets).T.contiguous()
        with torch.no_grad(): x.zero_()
        return loss
    model.vocab_parallel_cross_entropy = mutating_backend
    result = model.compute_language_model_loss(labels, logits)
    actual = torch.autograd.grad(result.mean(), logits)[0]
    ref = original.clone().requires_grad_()
    expected_loss = ce(ref, labels.T).mean() + .5e-4*ref.max(-1).values.square().mean()
    expected = torch.autograd.grad(expected_loss, ref)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('coefficient,training', [(0., True), (1e-4, False)])
def test_disabled_or_evaluation_keeps_original_ce(coefficient, training):
    model = minimal_model(coefficient)
    model.train(training)
    logits = torch.randn(3, 2, 11, requires_grad=True)
    labels = torch.randint(0, 11, (2, 3))
    model.vocab_parallel_cross_entropy = lambda x, y, group: ce(x, y).T
    with patch('rwkv_megatron.rwkv7_model.max_logit_stats', side_effect=AssertionError('must not wrap')):
        actual = model.compute_language_model_loss(labels, logits)
    expected = ce(logits, labels.T)
    assert torch.equal(actual, expected)
    a = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad(expected.sum(), logits)[0]
    assert torch.equal(a, b)


def test_yaml_and_cli_option_precedence(tmp_path, monkeypatch):
    path = tmp_path/'config.yaml'
    path.write_text('rwkv_l2wrap_coefficient: 0.0001\nrwkv_key_fusion: false\n')
    monkeypatch.setattr(sys, 'argv', ['training', '--rwkv-config', str(path)])
    parser = add_rwkv_arguments(argparse.ArgumentParser())
    args = parser.parse_args()
    assert args.rwkv_l2wrap_coefficient == 1e-4 and args.rwkv_key_fusion is False
    args = parser.parse_args(['--rwkv-l2wrap-coefficient', '0', '--rwkv-key-fusion'])
    assert args.rwkv_l2wrap_coefficient == 0 and args.rwkv_key_fusion is True
    path.write_text('language_model:\n  rwkv_l2wrap_coefficient: 0.0001\n  rwkv_key_fusion: true\n')
    args = add_rwkv_arguments(argparse.ArgumentParser()).parse_args()
    assert args.rwkv_l2wrap_coefficient == 1e-4 and args.rwkv_key_fusion is True


@pytest.mark.parametrize('value', ['-1', 'nan', 'inf'])
def test_invalid_coefficient_rejected(value, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['training'])
    parser = add_rwkv_arguments(argparse.ArgumentParser())
    with pytest.raises(SystemExit): parser.parse_args(['--rwkv-l2wrap-coefficient', value])


def test_yaml_key_fusion_requires_boolean(tmp_path, monkeypatch):
    path = tmp_path/'config.yaml'
    path.write_text('rwkv_key_fusion: "false"\n')
    monkeypatch.setattr(sys, 'argv', ['training', '--rwkv-config', str(path)])
    with pytest.raises(ValueError, match='YAML boolean'):
        add_rwkv_arguments(argparse.ArgumentParser())
