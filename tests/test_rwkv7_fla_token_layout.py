"""FLA point operators must receive training tokens on their time axis."""
from unittest.mock import patch

import torch

from rwkv_megatron.rwkv7_layer_specs import mix_inputs, update_key


def test_mix_keeps_batch_samples_and_gradient_with_one_or_many_batches():
    for batch in (1,3):
        x=torch.arange(4*batch*8,dtype=torch.float64).reshape(4,batch,8).requires_grad_()
        delta=(x.detach()/16).requires_grad_()
        coeff=torch.full((6,8),0.25,dtype=torch.float64,requires_grad=True)
        def point_op(value,shift,*weights):
            assert value.shape==(1,4*batch,8)
            return tuple(torch.addcmul(value,shift,w) for w in weights)
        with patch('rwkv_megatron.rwkv7_layer_specs.fused_addcmul_rwkv7',side_effect=point_op):
            actual=mix_inputs(x,delta,coeff.unbind(0))
        expected=tuple(torch.addcmul(x,delta,w) for w in coeff)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
        a=torch.autograd.grad(sum(t.sum() for t in actual),(x,delta,coeff))
        b=torch.autograd.grad(sum(t.sum() for t in expected),(x,delta,coeff))
        for first,second in zip(a,b):torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_key_uses_training_dispatch_without_mixing_token_or_hidden_axes():
    for batch in (1,3):
        key=torch.arange(4*batch*8,dtype=torch.float64).reshape(4,batch,8).requires_grad_()
        rate=torch.full_like(key,0.75,requires_grad=True)
        alpha=torch.full((8,),0.5,dtype=torch.float64,requires_grad=True)
        def point_op(k,a,ka):
            assert k.shape==(1,4*batch,8) and a.shape==k.shape
            return k*(1+(a-1)*ka)
        with patch('rwkv_megatron.rwkv7_layer_specs.fused_k_rwkv7',side_effect=point_op):
            actual=update_key(key,rate,alpha)
        expected=key*(1+(rate-1)*alpha)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        a=torch.autograd.grad(actual.sum(),(key,rate,alpha))
        b=torch.autograd.grad(expected.sum(),(key,rate,alpha))
        for first,second in zip(a,b):torch.testing.assert_close(first,second,rtol=0,atol=0)
