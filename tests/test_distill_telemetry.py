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
    assert not metrics['distill/student_ce_loss'].requires_grad
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
        logits[:, :, :5].reshape(-1, 5), labels.reshape(-1), reduction='mean'
    )
    assert torch.allclose(metrics['distill/student_ce_loss'][0] / metrics['distill/student_ce_loss'][1], expected)

    # A very large padded row would dominate a full-vocabulary softmax, but it
    # must not affect the reported effective-vocabulary CE.
    full_vocab = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 6), labels[:, :-1].reshape(-1), reduction='mean'
    )
    assert not torch.allclose(expected, full_vocab)


def test_monitor_matches_direct_kl_gradients_with_all_next_token_labels():
    from rwkv_megatron.online_teacher import kl_divergence_topk
    args = SimpleNamespace(rwkv_distill_temperature=2.0, rwkv_distill_logit_weight=1.0,
                           rwkv_distill_ce_weight=0.0, rwkv_distill_valid_vocab_size=7,
                           context_parallel_size=1, check_for_nan_in_loss_and_grad=False)
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(2, 4, 9, dtype=dtype, requires_grad=True)
        teacher = TeacherTopK(torch.randint(7, (2, 4, 3)), torch.randn(2, 4, 3))
        labels = torch.tensor([[1, 2, 3, 4], [2, -100, -100, -100]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
        with patch.object(pretrain, 'get_args', return_value=args):
            actual, count, metrics = pretrain.distillation_loss_func(mask, labels, (logits, teacher))
        reference = kl_divergence_topk(logits[:, :-1], TeacherTopK(
            teacher.token_ids[:, 1:], teacher.logprobs[:, 1:]), mask[:, :-1],
            temperature=2, valid_vocab_size=7) * count
        ga, = torch.autograd.grad(actual, logits, retain_graph=True)
        gr, = torch.autograd.grad(reference, logits)
        assert torch.equal(actual, reference)
        assert torch.equal(ga, gr)
        expected_ce = torch.nn.functional.cross_entropy(logits[mask.bool()][:, :7].float(), labels[mask.bool()])
        ce_pair = metrics['distill/student_ce_loss']
        assert ce_pair[1] == 5
        torch.testing.assert_close(ce_pair[0] / ce_pair[1], expected_ce)


def test_ppl_uses_aggregated_ce_not_average_microbatch_ppl():
    from rwkv_megatron.distill_metrics import StudentPerplexityCallback
    callback = StudentPerplexityCallback()
    ce = (torch.tensor(1.) * 2 + torch.tensor(3.) * 6) / 8
    context = SimpleNamespace(loss_dict={'distill/student_ce_loss': ce})
    callback.on_train_step_end(context)
    torch.testing.assert_close(context.loss_dict['distill/student_ppl'], ce.double().exp())
    assert not torch.isclose(context.loss_dict['distill/student_ppl'],
                             (torch.exp(torch.tensor(1.)) + torch.exp(torch.tensor(3.))).double() / 2)
