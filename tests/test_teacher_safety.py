"""Teacher targets are ready before replay; failed preparation never enters forward."""

import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from rwkv_megatron.online_teacher import TeacherPrefetchIterator, TeacherTopK
from rwkv_megatron.teacher_safety import TeacherStepBuffer, validate_teacher_safety


def rows(count=3):
    return [dict(tokens=torch.tensor([[i, i + 1]]), raw_prompts=[str(i)]) for i in range(count)]


class Teacher:
    def __init__(self):
        self.calls = 0

    def get_logits(self, tokens, **kwargs):
        self.calls += 1
        return TeacherTopK(tokens[..., None].int(), torch.zeros(*tokens.shape, 1))


def test_step_replay_has_exact_targets_without_another_teacher_request():
    source = rows()
    teacher = Teacher()
    buffer = TeacherStepBuffer(iter(source))
    buffer.prepare(2, teacher, time.monotonic() + 5, True)
    assert teacher.calls == 2
    for expected in source[:2]:
        actual = next(buffer)
        assert torch.equal(actual['teacher_token_ids'][..., 0], expected['tokens'])
    assert teacher.calls == 2
    with pytest.raises(RuntimeError, match='more microbatches'):
        next(buffer)
    buffer.prepare(1, teacher, time.monotonic() + 5, True)
    assert next(buffer)['raw_prompts'] == ['2']


def test_paired_pool_results_and_first_pipeline_stage_do_not_request_again():
    teacher = Teacher()
    source = rows(1)
    paired = dict(source[0], teacher_token_ids=torch.tensor([[[1], [2]]]),
                  teacher_logprobs=torch.zeros(1, 2, 1))
    for row, need_teacher in [(paired, True), (source[0], False)]:
        buffer = TeacherStepBuffer(iter([row]))
        buffer.prepare(1, teacher, time.monotonic() + 5, need_teacher)
        assert next(buffer) is row
    assert teacher.calls == 0


def test_async_future_is_resolved_during_preparation_and_wait_is_bounded():
    future = Future()
    row = dict(rows(1)[0], **{TeacherPrefetchIterator.future_key: future})
    buffer = TeacherStepBuffer(iter([row]))
    with pytest.raises(TimeoutError):
        buffer.prepare(1, Teacher(), time.monotonic() + 0.01, True)
    assert not buffer.rows
    future.set_result(Teacher().get_logits(row['tokens']))
    buffer = TeacherStepBuffer(iter([row]))
    buffer.prepare(1, Teacher(), time.monotonic() + 5, True)
    assert 'teacher_token_ids' in next(buffer)


def test_unconsumed_step_is_rejected_instead_of_discarding_samples():
    buffer = TeacherStepBuffer(iter(rows()))
    buffer.prepare(1, Teacher(), time.monotonic() + 5, False)
    with pytest.raises(RuntimeError, match='unconsumed'):
        buffer.prepare(1, Teacher(), time.monotonic() + 5, False)


@pytest.mark.parametrize('timeout', [float('inf'), float('nan'), -1, 0, 550])
def test_invalid_step_deadline_is_rejected_before_training(timeout):
    args = SimpleNamespace(rwkv_distill=True, rwkv_teacher_step_timeout=timeout,
                           distributed_timeout_minutes=10, dataloader_type='single',
                           virtual_pipeline_model_parallel_size=None)
    with pytest.raises(ValueError):
        validate_teacher_safety(args)
