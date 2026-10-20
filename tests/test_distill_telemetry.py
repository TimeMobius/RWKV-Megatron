from types import SimpleNamespace
from unittest.mock import patch

import torch

import pretrain_rwkv7 as pretrain
from rwkv_megatron.online_teacher import TeacherTopK


def test_telemetry_reporting_preserves_loss_and_gradients():
    args = SimpleNamespace(rwkv_distill_temperature=2.0, rwkv_distill_logit_weight=1.0,
                           rwkv_distill_ce_weight=0.0, context_parallel_size=1,
                           check_for_nan_in_loss_and_grad=True)
    logits = torch.randn(1, 4, 7, requires_grad=True)
    teacher = TeacherTopK(torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]]),
                         torch.log_softmax(torch.randn(1, 4, 2), -1))
    mask = torch.tensor([[1, 1, 0, 1]])
    labels = torch.tensor([[1, 2, 3, 4]])
    with patch.object(pretrain, 'get_args', return_value=args), patch('torch.distributed.get_rank', return_value=0):
        original, count, before = pretrain.distillation_loss_func(mask, labels, (logits, teacher))
        actual, actual_count, metrics = pretrain.distillation_loss_func(
            mask, labels, (logits, teacher), telemetry={'teacher/wait_seconds': 0.125, 'teacher/peak_active': 16})
    assert torch.equal(original, actual) and count.item() == actual_count.item() == 2
    assert torch.equal(before['lm loss'], metrics['lm loss'])
    assert torch.equal(metrics['lm loss'], metrics['distill/kl_loss'])
    assert metrics['teacher/wait_seconds'].tolist() == [0.125, 1.0]
    assert metrics['teacher/peak_active'].tolist() == [16.0, 1.0]
    ga, = torch.autograd.grad(original, logits, retain_graph=True)
    gb, = torch.autograd.grad(actual, logits)
    assert torch.equal(ga, gb)
