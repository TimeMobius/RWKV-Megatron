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
    assert 'distill/student_ce_loss' in metrics
    assert 'distill/student_ppl' in metrics
    ga, = torch.autograd.grad(original, logits, retain_graph=True)
    gb, = torch.autograd.grad(actual, logits)
    assert torch.equal(ga, gb)


def test_student_ce_monitor_excludes_padded_vocab_and_does_not_change_kl():
    args = SimpleNamespace(rwkv_distill_temperature=1.0, rwkv_distill_logit_weight=1.0,
                           rwkv_distill_ce_weight=0.0, rwkv_distill_valid_vocab_size=5,
                           context_parallel_size=1, check_for_nan_in_loss_and_grad=True)
    logits = torch.tensor([[[2.0, 0.0, -1.0, 0.5, 1.5, 100.0],
                           [0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]]], requires_grad=True)
    teacher = TeacherTopK(
        torch.tensor([[[0, 1], [1, 2], [2, 3]]]),
        torch.log_softmax(torch.tensor([[[0.0, -1.0], [0.0, -1.0], [0.0, -1.0]]]), -1),
    )
    labels = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(labels, dtype=torch.float32)
    with patch.object(pretrain, 'get_args', return_value=args), patch('torch.distributed.get_rank', return_value=0):
        _, _, metrics = pretrain.distillation_loss_func(mask, labels, (logits, teacher))
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :5].reshape(-1, 5), labels[:, :-1].reshape(-1), reduction='mean'
    )
    assert torch.allclose(metrics['distill/student_ce_loss'][0] / metrics['distill/student_ce_loss'][1], expected)
    assert torch.allclose(metrics['distill/student_ppl'][0], expected.exp())

    # A very large padded row would dominate a full-vocabulary softmax, but it
    # must not affect the reported effective-vocabulary CE.
    full_vocab = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 6), labels[:, :-1].reshape(-1), reduction='mean'
    )
    assert not torch.allclose(expected, full_vocab)
