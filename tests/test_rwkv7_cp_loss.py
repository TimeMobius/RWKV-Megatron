"""CP loss must match the global masked mean after DDP's CP averaging."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'pretrain_rwkv7.py'


def load_loss(cp_size, reduced_stats):
    definition = next(node for node in ast.parse(SOURCE.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'loss_func')
    calls = []
    group = object()

    def all_reduce(value, group=None):
        # c10d must see detached statistics, never differentiable tensors.
        assert not value.requires_grad
        calls.append(group)
        value.copy_(reduced_stats)

    namespace = {'torch': torch, 'get_args': lambda: SimpleNamespace(
        context_parallel_size=cp_size, check_for_nan_in_loss_and_grad=False),
        'mpu': SimpleNamespace(get_context_parallel_group=lambda: group)}
    # A local proxy keeps this CPU test independent of distributed init.
    namespace['torch'] = SimpleNamespace(
        Tensor=torch.Tensor, cat=torch.cat, sum=torch.sum, int=torch.int,
        distributed=SimpleNamespace(all_reduce=all_reduce))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['loss_func'], calls, group


@pytest.mark.parametrize('masks', [
    [[1, 1, 1, 1], [1, 1, 1, 1]],
    [[1, 0, 0, 0], [1, 1, 1, 1]],
    [[0, 0, 0, 0], [1, 0, 1, 1]],
    [[0, 0, 0, 0], [0, 0, 0, 0]],
])
def test_cp_loss_matches_global_masked_mean_and_only_reduces_detached_stats(masks):
    masks = torch.tensor(masks, dtype=torch.float32)
    values = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    stats = torch.tensor([(values*masks).sum(), masks.sum()])
    baseline = values.clone().requires_grad_(True)
    cp1, calls, _ = load_loss(1, stats)
    loss, count, report = cp1(masks.flatten(), baseline.flatten())
    (loss/count.clamp_min(1)).backward()
    assert not calls
    assert torch.equal(report['lm loss'], stats)
    local_gradients = []
    for rank in range(2):
        local = values[rank].clone().requires_grad_(True)
        cp2, calls, group = load_loss(2, stats)
        loss, count, report = cp2(masks[rank], local)
        assert loss.item() == 2*stats[0].item()
        assert count.item() == stats[1].item()
        assert torch.equal(report['lm loss'], stats)
        assert not report['lm loss'].requires_grad
        assert calls == [group]
        (loss/count.clamp_min(1)).backward()
        # Production DDP averages across CP replicas (factor 1/CP).
        local_gradients.append(local.grad/2)
    torch.testing.assert_close(torch.stack(local_gradients), baseline.grad, rtol=0, atol=0)
