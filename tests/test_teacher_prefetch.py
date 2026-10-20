"""Concurrent teacher requests must preserve sample and gradient identity."""

import threading
import argparse
import sys
from collections import Counter
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rwkv_megatron.online_teacher import TeacherPrefetchIterator, TeacherTopK, kl_divergence_topk


def batches(count):
    return [{"tokens": torch.tensor([[i, i + 1, i + 2]]), "sample_id": i} for i in range(count)]


class FrozenTeacher:
    def get_logits(self, tokens):
        ids = (tokens.unsqueeze(-1) + torch.arange(3)) % 17
        lp = torch.log_softmax(ids.float(), dim=-1)
        return TeacherTopK(ids.int(), lp)


def test_parallel_out_of_order_completion_stays_paired_and_bounded():
    release_first = threading.Event()
    third_done = threading.Event()
    entered = threading.Barrier(3)

    class Teacher(FrozenTeacher):
        def get_logits(self, tokens):
            i = tokens[0, 0].item()
            if i < 3:
                entered.wait(timeout=5)
            if i == 0:
                assert release_first.wait(timeout=5)
            if i == 2:
                third_done.set()
            return super().get_logits(tokens)

    iterator = TeacherPrefetchIterator(batches(8), Teacher(), num_workers=3, lookahead=4)
    try:
        first = next(iterator)
        assert third_done.wait(timeout=5)
        assert not first[iterator.future_key].done()
        stats = iterator.get_stats()
        assert stats["peak_active"] == 3
        assert stats["submitted"] == 4 and stats["buffered"] == 3
        release_first.set()
        output = [first, *list(iterator)]
        assert [x["sample_id"] for x in output] == list(range(8))
        for batch in output:
            result = batch[iterator.future_key].result(timeout=5)
            assert result.token_ids[0, 0, 0].item() == batch["sample_id"]
        assert iterator.get_stats()["submitted"] == 8
    finally:
        release_first.set()
        iterator.close()


def test_dynamic_concurrency_reacts_to_latency_and_rate_limits():
    iterator = TeacherPrefetchIterator(batches(1), FrozenTeacher(), num_workers=4,
                                       lookahead=1, min_workers=1, latency_target_seconds=2.0)
    try:
        assert iterator.get_stats()["target_workers"] == 4
        iterator._observe_request({"latency_seconds": 3.0, "retries": 0, "rate_limits": 0})
        assert iterator.get_stats()["target_workers"] == 3
        iterator._observe_request({"latency_seconds": 0.1, "retries": 0, "rate_limits": 0})
        assert iterator.get_stats()["target_workers"] == 4
        iterator._observe_request({"latency_seconds": 0.1, "retries": 1, "rate_limits": 1})
        assert iterator.get_stats()["target_workers"] == 2
        assert iterator.get_stats()["observed_rate_limits"] == 1
    finally:
        iterator.close()


def test_frozen_teacher_sync_async_loss_and_gradient_are_identical():
    torch.manual_seed(17)
    teacher = FrozenTeacher()
    data = batches(9)
    logits = [torch.randn(1, 2, 17) for _ in data]
    iterator = TeacherPrefetchIterator(data, teacher, num_workers=4, lookahead=5)
    try:
        for i, batch in enumerate(iterator):
            synchronous = teacher.get_logits(data[i]["tokens"])
            asynchronous = batch[iterator.future_key].result(timeout=5)
            a, b = [x.clone().requires_grad_() for x in (logits[i], logits[i])]

            def loss(student, result):
                aligned = TeacherTopK(result.token_ids[:, 1:], result.logprobs[:, 1:])
                return kl_divergence_topk(student, aligned, torch.tensor([[1, i % 2]]), 2.0)

            la, lb = loss(a, synchronous), loss(b, asynchronous)
            ga, = torch.autograd.grad(la, a)
            gb, = torch.autograd.grad(lb, b)
            assert torch.equal(la, lb)
            assert torch.equal(ga, gb)
    finally:
        iterator.close()


def test_http_failure_propagates_without_duplicate_request_or_skipped_batch():
    calls = Counter()
    lock = threading.Lock()

    class Teacher(FrozenTeacher):
        def get_logits(self, tokens):
            i = tokens[0, 0].item()
            with lock:
                calls[i] += 1
            if i == 1:
                raise RuntimeError("teacher HTTP failure")
            return super().get_logits(tokens)

    iterator = TeacherPrefetchIterator(batches(3), Teacher(), num_workers=2, lookahead=3)
    try:
        next(iterator)[iterator.future_key].result(timeout=5)
        with pytest.raises(RuntimeError, match="teacher HTTP failure"):
            next(iterator)[iterator.future_key].result(timeout=5)
        third = next(iterator)
        assert third["sample_id"] == 2
        third[iterator.future_key].result(timeout=5)
    finally:
        iterator.close()
    assert calls == Counter({0: 1, 1: 1, 2: 1})
    assert iterator.get_stats()["failed"] == 1


def test_checkpoint_resume_uses_consumed_samples_not_prefetched_cursor():
    from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode
    from megatron.training.datasets.data_samplers import MegatronPretrainingSampler

    data = batches(20)

    def source(consumed):
        sampler = MegatronPretrainingSampler(20, consumed, 1, 0, 1)
        return (data[indices[0]] for indices in sampler)

    pending = TeacherPrefetchIterator(source(0), FrozenTeacher(), num_workers=2, lookahead=8)
    rerun = RerunDataIterator(pending)
    machine = SimpleNamespace(get_mode=lambda: RerunMode.DISABLED)
    try:
        with patch("megatron.core.rerun_state_machine.get_rerun_state_machine", return_value=machine):
            consumed = [next(rerun)["sample_id"] for _ in range(3)]
        assert pending.get_stats()["submitted"] > len(consumed)
        assert rerun.state_dict()["saved_microbatches"] == []
    finally:
        pending.close()
    resumed = TeacherPrefetchIterator(source(len(consumed)), FrozenTeacher(), num_workers=2, lookahead=8)
    try:
        assert [batch["sample_id"] for batch in resumed] == list(range(3, 20))
    finally:
        resumed.close()


def test_close_cancels_queue_and_drains_running_calls():
    entered = threading.Event()
    release = threading.Event()

    class Teacher(FrozenTeacher):
        def get_logits(self, tokens):
            entered.set()
            assert release.wait(timeout=5)
            return super().get_logits(tokens)

    iterator = TeacherPrefetchIterator(batches(20), Teacher(), num_workers=1, lookahead=10)
    first = next(iterator)
    assert entered.wait(timeout=5)
    queued = [future for _, future in iterator._pending]
    closer = threading.Thread(target=iterator.close)
    closer.start()
    try:
        # Pending futures must be cancelled while the first HTTP call is live.
        with pytest.raises(CancelledError):
            queued[-1].result(timeout=5)
    finally:
        release.set()
        closer.join(timeout=5)
    assert not closer.is_alive()
    assert first[iterator.future_key].done()
    assert iterator.get_stats()["active"] == 0
    assert iterator.get_stats()["completed"] == 1
    iterator.close()
    assert list(iterator) == []


def test_prefetch_workers_cannot_mutate_batch_tokens():
    class Teacher(FrozenTeacher):
        def get_logits(self, tokens):
            tokens.fill_(16)
            return super().get_logits(tokens)

    data = batches(1)
    iterator = TeacherPrefetchIterator(data, Teacher(), num_workers=1, lookahead=1)
    try:
        batch = next(iterator)
        batch[iterator.future_key].result(timeout=5)
        assert torch.equal(batch["tokens"], torch.tensor([[0, 1, 2]]))
    finally:
        iterator.close()


def test_async_config_validation_and_cli_overrides(tmp_path, monkeypatch):
    from rwkv_megatron.arguments import add_rwkv_arguments

    config = tmp_path / "async.yaml"
    config.write_text("rwkv_teacher_async_prefetch: true\nrwkv_teacher_prefetch_workers: 16\n"
                      "rwkv_teacher_prefetch_lookahead: 32\n")
    monkeypatch.setattr(sys, "argv", ["train", "--rwkv-config", str(config)])
    parser = add_rwkv_arguments(argparse.ArgumentParser())
    args = parser.parse_args()
    assert args.rwkv_teacher_async_prefetch
    assert args.rwkv_teacher_prefetch_workers == 16
    assert args.rwkv_teacher_prefetch_lookahead == 32
    args = parser.parse_args(["--no-rwkv-teacher-async-prefetch", "--rwkv-teacher-prefetch-workers", "4"])
    assert not args.rwkv_teacher_async_prefetch
    assert args.rwkv_teacher_prefetch_workers == 4
    with pytest.raises(SystemExit):
        parser.parse_args(["--rwkv-teacher-prefetch-workers", "0"])
    config.write_text("rwkv_teacher_prefetch_lookahead: 0\n")
    with pytest.raises(ValueError, match="positive integer"):
        add_rwkv_arguments(argparse.ArgumentParser())


def test_planned_exit_does_not_request_unused_prompts():
    iterator = TeacherPrefetchIterator(batches(20), FrozenTeacher(), lookahead=32, max_batches=3)
    try:
        output = list(iterator)
        assert [x["sample_id"] for x in output] == [0, 1, 2]
        assert iterator.get_stats()["submitted"] == 3
    finally:
        iterator.close()


def test_close_releases_multiprocess_dataloader_and_joins_workers():
    import weakref
    from torch.utils.data import DataLoader

    source = iter(DataLoader(batches(12), batch_size=None, num_workers=2))
    workers = list(source._workers)
    reference = weakref.ref(source)
    iterator = TeacherPrefetchIterator(source, FrozenTeacher(), num_workers=2, lookahead=4)
    del source
    try:
        next(iterator)[iterator.future_key].result(timeout=5)
    finally:
        iterator.close()
    assert reference() is None
    assert all(not worker.is_alive() for worker in workers)
