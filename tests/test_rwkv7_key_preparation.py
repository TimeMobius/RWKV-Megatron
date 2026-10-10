"""Independent key formula/gradient checks and compiler compatibility."""
from unittest.mock import patch

import pytest
import torch

import rwkv_megatron.key_preparation as impl


def formula(key, learn, beta, alpha, head_size):
    grouped = (key * beta).reshape(*key.shape[:-1], -1, head_size)
    norm = grouped.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    kk = (grouped / norm).reshape_as(key)
    return key * (1 + (learn - 1) * alpha), -kk, learn * kk


def inputs(device='cpu', dtype=torch.float64, batch=2):
    generator = torch.Generator().manual_seed(17)
    shapes = [(3, batch, 128)] * 2 + [(128,)] * 2
    return [torch.rand(s, generator=generator).to(device=device, dtype=dtype).requires_grad_() for s in shapes]


def test_formula_head_and_batch_axes_and_gradients():
    for batch in (1, 3):
        tensors = inputs(batch=batch)
        actual = impl.prepare_key(*tensors, 64)
        expected = formula(*tensors, 64)
        dy = [torch.randn_like(t) for t in actual]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        a = torch.autograd.grad(actual, tensors, dy)
        b = torch.autograd.grad(expected, tensors, dy)
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10)
    assert torch.autograd.gradcheck(lambda *x: impl.prepare_key(*x, 64), tuple(inputs()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for compiled path')
def test_compiled_bf16_output_and_backward_against_independent_formula(monkeypatch):
    monkeypatch.setenv('RWKV7_KEY_FUSION', '1')
    monkeypatch.setattr(impl, '_compiled_key', None)
    monkeypatch.setattr(impl, '_fusion_unavailable', False)
    tensors = inputs('cuda', torch.bfloat16)
    reference = [t.detach().double().requires_grad_() for t in tensors]
    expected = formula(*reference, 64)
    actual = impl.prepare_key(*tensors, 64)
    dy = [torch.randn_like(t) for t in actual]
    grads = torch.autograd.grad(actual, tensors, dy)
    refgrads = torch.autograd.grad(expected, reference, [t.double() for t in dy])
    assert not impl._fusion_unavailable
    for a, b in zip(actual, expected):
        assert a.dtype == torch.bfloat16 and a.shape == b.shape
        assert (a.double()-b).norm()/b.norm() < 0.0025
    for a, b in zip(grads, refgrads):
        assert torch.isfinite(a).all()
        assert (a.double()-b).norm()/b.norm() < 0.004


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for compiler compatibility')
def test_compile_unavailable_falls_back_once_and_explicit_disable(monkeypatch):
    tensors = inputs('cuda', torch.bfloat16)
    monkeypatch.setenv('RWKV7_KEY_FUSION', '1')
    monkeypatch.setattr(impl, '_compiled_key', None)
    monkeypatch.setattr(impl, '_fusion_unavailable', False)
    with patch.object(torch, 'compile', side_effect=ImportError('compiler missing')) as compile_mock:
        with pytest.warns(RuntimeWarning, match='Torch FP32'):
            actual = impl.prepare_key(*tensors, 64)
        impl.prepare_key(*tensors, 64)
        assert compile_mock.call_count == 1
    expected = impl.key_fp32(*tensors, 64)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    monkeypatch.setattr(impl, '_fusion_unavailable', False)
    monkeypatch.setenv('RWKV7_KEY_FUSION', '0')
    with patch.object(torch, 'compile', side_effect=AssertionError('must not compile')):
        outputs = impl.prepare_key(*tensors, 64)
        grads = torch.autograd.grad(outputs, tensors, [torch.ones_like(t) for t in outputs])
        assert all(torch.isfinite(t).all() for t in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for error propagation')
def test_runtime_errors_are_not_hidden_by_fallback(monkeypatch):
    monkeypatch.setenv('RWKV7_KEY_FUSION', '1')
    monkeypatch.setattr(impl, '_fusion_unavailable', False)
    with patch.object(impl, '_compiled_key', side_effect=RuntimeError('invalid shape')):
        with pytest.raises(RuntimeError, match='invalid shape'):
            impl.prepare_key(*inputs('cuda', torch.bfloat16), 64)
    assert not impl._fusion_unavailable


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for backend error handling')
def test_backend_compile_failure_uses_fp32_fallback(monkeypatch):
    monkeypatch.setenv('RWKV7_KEY_FUSION', '1')
    monkeypatch.setattr(impl, '_fusion_unavailable', False)
    error = torch._dynamo.exc.BackendCompilerFailed(lambda: None, RuntimeError('compiler failed'), None)
    tensors = inputs('cuda', torch.bfloat16)
    with patch.object(impl, '_compiled_key', side_effect=error):
        with pytest.warns(RuntimeWarning, match='BackendCompilerFailed'):
            outputs = impl.prepare_key(*tensors, 64)
        gradients = torch.autograd.grad(outputs, tensors, [torch.ones_like(t) for t in outputs])
    assert impl._fusion_unavailable and all(torch.isfinite(t).all() for t in gradients)
