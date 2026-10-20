"""Pool isolation, bounded buffering, failures, and exact KL/update identity."""

import argparse
import copy
import multiprocessing
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from rwkv_megatron.online_teacher import TeacherPrefetchIterator, TeacherTopK, kl_divergence_topk
from rwkv_megatron.teacher_pool import RemoteTeacherPoolClient, TeacherPool, make_pool_server


class Tokenizer:
    def get_vocab(self):
        return {str(i): i for i in range(17)}


class FrozenTeacher:
    server_url = "http://frozen"
    model_name = "test"
    top_k = 3
    request_seed = 1234
    request_temperature = 1.0

    def _get_tokenizer(self):
        return Tokenizer()

    def get_effective_vocab_size(self):
        return 17

    def get_logits(self, tokens, attention_mask=None, prompts=None):
        if prompts == ["slow"]:
            time.sleep(0.3)
            prompts = None
        if int(tokens[0, 0]) == 99:
            raise ValueError("frozen upstream error")
        if prompts is not None and prompts != [f"sample-{int(tokens[0, 0])}"]:
            raise ValueError("prompt does not match sample")
        ids = (tokens.unsqueeze(-1) + torch.arange(3)) % 17
        lp = torch.log_softmax(ids.float(), -1)
        if attention_mask is not None:
            ids = ids.masked_fill(~attention_mask.bool().unsqueeze(-1), -1)
            lp = lp.masked_fill(~attention_mask.bool().unsqueeze(-1), -torch.inf)
        return TeacherTopK(ids.int(), lp)

    def get_last_request_metrics(self):
        return dict(latency_seconds=0.01, retries=0, rate_limits=0, cache_hit=False)

    def get_stats(self):
        return {}


def run_server(connection):
    torch.set_num_threads(1)
    pool = TeacherPool(FrozenTeacher(), workers=3, queue_size=2, max_results=2)
    server = make_pool_server(pool, port=0)
    connection.send(server.address[1])
    connection.close()
    server.serve_forever()


@pytest.fixture(scope="module")
def pool_port():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=run_server, args=(writer,))
    process.start()
    writer.close()
    try:
        assert reader.poll(30), "spawned teacher pool did not start"
        yield reader.recv()
    finally:
        reader.close()
        process.terminate()
        process.join(10)
        assert not process.is_alive()


def remote(port, **kwargs):
    return RemoteTeacherPoolClient(FrozenTeacher(), port=port, **kwargs)


def submit(pool, key, value=0):
    tokens = torch.tensor([[value, value + 1, value + 2]])
    return pool.submit(key, tokens.numpy().tobytes(), None, tuple(tokens.shape), None)


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_separate_process_multiple_clients_and_threads_preserve_sample_identity(pool_port):
    clients = [remote(pool_port), remote(pool_port)]
    start = clients[0].get_stats()["pool_consumed"]

    def fetch(i):
        client = clients[i % 2]
        tokens = torch.tensor([[i, i + 1, i + 2]])
        expected = FrozenTeacher().get_logits(tokens)
        # A pool client must never fall back to the local teacher.
        client.local_client.get_logits = lambda *a, **k: pytest.fail("direct teacher fallback")
        actual = client.get_logits(tokens, prompts=[f"sample-{i}"])
        assert torch.equal(actual.token_ids, expected.token_ids)
        assert torch.equal(actual.logprobs, expected.logprobs)
        return client.get_last_request_metrics()

    with ThreadPoolExecutor(max_workers=6) as executor:
        metrics = list(executor.map(fetch, range(12)))
    assert all(m["retries"] == 0 for m in metrics)
    assert clients[0].get_stats()["pool_consumed"] - start == 12
    assert clients[0].get_stats()["pool_retained"] == 0
    assert len(clients[0].get_training_stats()) == 8


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pool_transport_loss_and_gradients_are_bitwise_identical(pool_port, dtype):
    tokens = torch.tensor([[2, 3, 4], [5, 6, 7]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    expected = FrozenTeacher().get_logits(tokens, attention_mask=mask)
    actual = remote(pool_port).get_logits(tokens, attention_mask=mask)
    torch.manual_seed(71)
    logits = torch.randn(2, 2, 20).to(dtype)
    outcomes = []
    for teacher in (expected, actual):
        student = logits.clone().requires_grad_()
        aligned = TeacherTopK(teacher.token_ids[:, 1:], teacher.logprobs[:, 1:])
        loss = kl_divergence_topk(student, aligned, mask[:, 1:], valid_vocab_size=17)
        grad, = torch.autograd.grad(loss, student)
        outcomes.append((loss, grad))
    assert torch.equal(outcomes[0][0], outcomes[1][0])
    assert torch.equal(outcomes[0][1], outcomes[1][1])
    assert outcomes[1][1][..., 17:].count_nonzero() == 0


def test_remote_prefetch_updates_and_consumed_cursor_resume_match_direct_training(pool_port):
    torch.manual_seed(21)
    initial = torch.nn.Embedding(17, 20).state_dict()
    data = [{"tokens": torch.tensor([[i, i + 1, i + 2]])} for i in range(7)]

    def run(client, count=7, saved=None):
        model = torch.nn.Embedding(17, 20)
        model.load_state_dict(initial if saved is None else saved[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        consumed = 0 if saved is None else 3
        if saved is not None:
            optimizer.load_state_dict(saved[1])
        iterator = TeacherPrefetchIterator(data[consumed:], client, num_workers=3, lookahead=4)
        history = []
        try:
            for _ in range(count):
                batch = next(iterator)
                teacher = batch[iterator.future_key].result(timeout=10)
                logits = model(batch["tokens"])[:, :-1]
                loss = kl_divergence_topk(logits, TeacherTopK(teacher.token_ids[:, 1:], teacher.logprobs[:, 1:]),
                                          torch.ones(1, 2), valid_vocab_size=17)
                loss.backward()
                history.append((loss.item(), model.weight.grad.clone()))
                optimizer.step()
                optimizer.zero_grad()
        finally:
            iterator.close()
        return history, (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))

    direct, direct_state = run(FrozenTeacher())
    pooled, pooled_state = run(remote(pool_port))
    first, checkpoint = run(remote(pool_port), count=3)
    resumed, resumed_state = run(remote(pool_port), count=4, saved=checkpoint)
    for a, b, c in zip(direct, pooled, first + resumed):
        assert a[0] == b[0] == c[0]
        assert torch.equal(a[1], b[1]) and torch.equal(a[1], c[1])
    assert torch.equal(direct_state[0]["weight"], pooled_state[0]["weight"])
    assert torch.equal(direct_state[0]["weight"], resumed_state[0]["weight"])


def test_remote_failure_is_not_a_zero_loss_or_fallback(pool_port):
    client = remote(pool_port)
    with pytest.raises(RuntimeError, match="frozen upstream error"):
        client.get_logits(torch.tensor([[99, 1, 2]]))
    with pytest.raises(RuntimeError, match="prompt does not match"):
        client.get_logits(torch.tensor([[1, 2, 3]]), prompts=["wrong"])
    assert client.get_stats()["pool_retained"] == 0


def test_remote_timeout_cancels_result_without_an_upstream_fallback(pool_port):
    client = remote(pool_port, timeout=0.1)
    before = client.get_stats()["pool_cancelled"]
    with pytest.raises(TimeoutError, match="result timed out"):
        client.get_logits(torch.tensor([[1, 2, 3]]), prompts=["slow"])
    assert client.get_stats()["pool_cancelled"] == before + 1
    wait_for(lambda: client.get_stats()["pool_active"] == 0)
    assert client.get_stats()["pool_retained"] == 0


def test_remote_backpressure_waits_until_an_unread_result_is_consumed(pool_port):
    client = remote(pool_port, timeout=5)
    assert submit(client._pool, "held-1", 1)
    assert submit(client._pool, "held-2", 2)
    try:
        wait_for(lambda: client.get_stats()["pool_ready"] == 2)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(client.get_logits, torch.tensor([[3, 4, 5]]))
            wait_for(lambda: client.get_stats()["backpressure_waits"] > 0)
            assert not future.done()
            assert client._pool.get("held-1", 1)["ok"]
            assert future.result(timeout=5).token_ids[0, 0, 0] == 3
        assert client._pool.get("held-2", 1)["ok"]
    finally:
        client._pool.cancel("held-1")
        client._pool.cancel("held-2")


@pytest.mark.parametrize("field,value", [("model_name", "wrong"), ("top_k", 2),
                                        ("request_seed", 0), ("request_temperature", 2.0),
                                        ("server_url", "http://other")])
def test_incompatible_teacher_configuration_is_rejected(pool_port, field, value):
    local = FrozenTeacher()
    setattr(local, field, value)
    with pytest.raises(ValueError, match="configuration mismatch"):
        RemoteTeacherPoolClient(local, port=pool_port)


def test_incompatible_vocab_and_bad_authentication_are_rejected(pool_port):
    local = FrozenTeacher()
    local._get_tokenizer = lambda: SimpleNamespace(get_vocab=lambda: {"different": 0})
    with pytest.raises(ValueError, match="vocab_sha256"):
        RemoteTeacherPoolClient(local, port=pool_port)
    with pytest.raises(multiprocessing.AuthenticationError):
        remote(pool_port, authkey="wrong")


def test_full_queue_backpressures_without_losing_ready_results():
    entered, release = threading.Event(), threading.Event()

    class BlockingTeacher(FrozenTeacher):
        def get_logits(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().get_logits(*args, **kwargs)

    pool = TeacherPool(BlockingTeacher(), workers=1, queue_size=1, max_results=2)
    try:
        assert submit(pool, "first", 1)
        assert entered.wait(5)
        assert submit(pool, "second", 2)
        assert not submit(pool, "third", 3)
        assert submit(pool, "first", 1)  # idempotent submission
        with pytest.raises(ValueError, match="reused"):
            submit(pool, "first", 4)
        release.set()
        wait_for(lambda: pool.get_stats()["ready"] == 2)
        assert not submit(pool, "third", 3)  # even when input queue is empty
        assert pool.get("first")["ok"]
        assert submit(pool, "third", 3)
        assert pool.get("second", 5)["ok"]
        assert pool.get("third", 5)["ok"]
        assert pool.get_stats()["retained"] == 0
    finally:
        release.set()
        pool.close()


def test_orphan_result_ttl_and_cancellation_release_capacity():
    pool = TeacherPool(FrozenTeacher(), workers=1, max_results=1, result_ttl=0.05)
    try:
        assert submit(pool, "orphan")
        wait_for(lambda: pool.get_stats()["expired"] == 1)
        with pytest.raises(RuntimeError, match="expired"):
            pool.get("orphan")
        assert submit(pool, "cancel")
        pool.cancel("cancel")
        assert pool.get_stats()["cancelled"] == 1
        assert submit(pool, "next")
        assert pool.get("next", 5)["ok"]
    finally:
        pool.close()


def test_global_concurrency_reacts_to_rate_limits():
    class LimitedTeacher(FrozenTeacher):
        def get_last_request_metrics(self):
            return dict(latency_seconds=0.1, retries=1, rate_limits=1)

    pool = TeacherPool(LimitedTeacher(), workers=4)
    try:
        submit(pool, "limited")
        assert pool.get("limited", 5)["ok"]
        assert pool.get_stats()["target_workers"] == 2
    finally:
        pool.close()


def test_cancelling_queued_work_does_not_issue_an_upstream_request():
    entered, release = threading.Event(), threading.Event()
    calls = []

    class BlockingTeacher(FrozenTeacher):
        def get_logits(self, tokens, **kwargs):
            calls.append(int(tokens[0, 0]))
            entered.set()
            assert release.wait(5)
            return super().get_logits(tokens, **kwargs)

    pool = TeacherPool(BlockingTeacher(), workers=1)
    try:
        submit(pool, "running", 1)
        assert entered.wait(5)
        submit(pool, "queued", 2)
        pool.cancel("queued")
        release.set()
        assert pool.get("running", 5)["ok"]
        assert calls == [1]
        assert pool.get_stats()["queued"] == 0
    finally:
        release.set()
        pool.close()


def test_server_death_is_visible_and_never_falls_back_to_direct_teacher():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=run_server, args=(writer,))
    process.start()
    writer.close()
    try:
        assert reader.poll(30)
        client = remote(reader.recv())
        client.local_client.get_logits = lambda *a, **k: pytest.fail("direct fallback after pool death")
        process.terminate()
        process.join(10)
        with pytest.raises((EOFError, OSError)):
            client.get_logits(torch.tensor([[1, 2, 3]]))
    finally:
        reader.close()
        if process.is_alive():
            process.terminate()
        process.join(10)


def test_pool_config_and_factory_selection(tmp_path, monkeypatch):
    from rwkv_megatron.arguments import add_rwkv_arguments
    from rwkv_megatron.online_teacher import OnlineTeacherClient, build_teacher_client

    config = tmp_path / "pool.yaml"
    config.write_text("rwkv_teacher_pool: true\nrwkv_teacher_pool_port: 50057\n"
                      "rwkv_teacher_tokenizer: unused\nrwkv_teacher_seed: 1234\n")
    monkeypatch.setattr(sys, "argv", ["train", "--rwkv-config", str(config)])
    parser = add_rwkv_arguments(argparse.ArgumentParser())
    args = parser.parse_args()
    assert args.rwkv_teacher_pool and args.rwkv_teacher_pool_port == 50057
    assert not parser.parse_args(["--no-rwkv-teacher-pool"]).rwkv_teacher_pool
    assert isinstance(build_teacher_client(args, use_pool=False), OnlineTeacherClient)
    captured = []
    monkeypatch.setattr("rwkv_megatron.teacher_pool.RemoteTeacherPoolClient",
                        lambda client, **kwargs: captured.append((client, kwargs)))
    build_teacher_client(args)
    assert captured[0][0].request_seed == 1234
    assert captured[0][0]._api_key == "sk-distill"
    assert captured[0][1]["port"] == 50057
    for key, value in [("rwkv_teacher_pool", "yes"), ("rwkv_teacher_pool_queue_size", 0),
                       ("rwkv_teacher_pool_timeout", float("nan")), ("rwkv_teacher_pool_port", 65536)]:
        import yaml

        config.write_text(yaml.safe_dump({key: value}))
        with pytest.raises(ValueError):
            add_rwkv_arguments(argparse.ArgumentParser())
