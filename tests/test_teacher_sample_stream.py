"""Pool-side replacement, background overlap, shared readers and journal replay."""

import copy
import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from rwkv_megatron.jsonl_dataset import JsonlTextDataset
from rwkv_megatron.online_teacher import TeacherSampleError, TeacherTopK, teacher_request_deadline
from rwkv_megatron.teacher_pool import (
    RemoteTeacherPoolClient, TeacherPool, _LocalPoolRequestClient, make_pool_server,
)
from rwkv_megatron.teacher_replacement import selection_signature
from rwkv_megatron.teacher_sample_stream import PoolValidatedDataset


class Tokenizer:
    def get_vocab(self):
        return {chr(i + 96): i for i in range(1, 27)}

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) - 96 for char in text])


class Teacher:
    server_url = "http://fixture"
    model_name = "fixture"
    top_k = 2
    request_seed = 1234
    request_temperature = 1.0
    prompt_mode = "text"

    def _get_tokenizer(self):
        return Tokenizer()

    def get_effective_vocab_size(self):
        return 32

    def get_logits(self, tokens, prompts=None, **kwargs):
        if prompts[0] in ("bb", "dd"):
            raise TeacherSampleError("injected position mismatch")
        if prompts[0] == "zz":
            raise TimeoutError("service offline")
        return TeacherTopK(tokens[..., None].int() + torch.arange(2),
                           torch.zeros(*tokens.shape, 2))

    def get_last_request_metrics(self):
        return dict(latency_seconds=0.01)

    def get_stats(self):
        return {}


def run_server(connection):
    torch.set_num_threads(1)
    pool = TeacherPool(Teacher(), workers=3, max_results=16)
    server = make_pool_server(pool, port=0)
    connection.send(server.address[1])
    connection.close()
    server.serve_forever()


@pytest.fixture
def pool_port():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=run_server, args=(writer,))
    process.start()
    writer.close()
    try:
        assert reader.poll(30)
        yield reader.recv()
    finally:
        reader.close()
        process.terminate()
        process.join(10)
        assert not process.is_alive()


def source_spec(tmp_path, prompts=None, count=12):
    path = tmp_path / "train.jsonl"
    if not path.exists():
        path.write_text("".join(json.dumps(dict(text=s)) + "\n" for s in (
            prompts or ["bb", "aa", "dd", "cc", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll", "mm", "nn", "oo"])))
    source = JsonlTextDataset([path], Tokenizer(), 4, target_samples=count)
    args = SimpleNamespace(rwkv_jsonl_max_records=0, rwkv_teacher_model="fixture",
                           rwkv_teacher_url="http://fixture", rwkv_teacher_top_k=2,
                           rwkv_teacher_seed=1234, rwkv_teacher_prompt_mode="text")
    return source, selection_signature(source, args, "train")


def dataset(tmp_path, client, lookahead=3, start_index=0, **kwargs):
    source, signature = source_spec(tmp_path, **kwargs)
    return PoolValidatedDataset(source, client, tmp_path / "selection", signature, lookahead, start_index)


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_separate_process_shares_paired_results_and_replacements_across_pp_dp(pool_port, tmp_path):
    clients = [RemoteTeacherPoolClient(Teacher(), port=pool_port) for _ in range(4)]
    readers = [dataset(tmp_path, c) for c in clients]
    for client in clients:
        client.local_client.get_logits = lambda *a, **k: pytest.fail("student HTTP request")
        client.local_client._get_tokenizer = lambda: pytest.fail("student tokenization")
    try:
        def read(reader):
            rows = [reader[i] for i in range(6)]
            for row in rows:
                assert torch.equal(row["teacher_token_ids"][..., 0], row["tokens"])
            return [row["raw_prompts"] for row in rows]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(read, readers))
        assert all(r == ["aa", "cc", "ee", "ff", "gg", "hh"] for r in results)
        assert readers[0].rejected_count() == 2
        stats = clients[0].sample_stream_stats(readers[0].key)
        assert stats["ready"] <= stats["capacity"]
        assert clients[0].get_stats()["pool_sample_streams"] == 1
        records = [json.loads(line) for line in (tmp_path / "selection/rejected-samples.jsonl").read_text().splitlines()]
        assert sorted(row["source_index"] for row in records) == [0, 2]
        # All four readers share inference results; no per-reader HTTP replay.
        assert clients[0].get_stats()["pool_submitted"] <= 14
    finally:
        for reader in readers:
            reader.close()


def test_background_slow_future_sample_does_not_block_current_ready_sample(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Blocking(Teacher):
        def get_logits(self, tokens, prompts=None, **kwargs):
            if prompts == ["cc"]:
                entered.set()
                assert release.wait(10)
            return super().get_logits(tokens, prompts=prompts, **kwargs)

    pool = TeacherPool(Blocking(), workers=2)
    client = _LocalPoolRequestClient(pool)
    reader = dataset(tmp_path, client, lookahead=2, prompts=["aa", "cc", "ee"], count=3)
    try:
        # Source 1 stays blocked while source 0 becomes usable. There must be
        # no main-thread "fill all lookahead batches before returning" wait.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(reader.__getitem__, 0)
            assert entered.wait(5)
            assert future.result(timeout=2)["raw_prompts"] == "aa"
        assert not release.is_set()
        assert reader[0]["raw_prompts"] == "aa"
        release.set()
        wait_for(lambda: client.sample_stream_stats(reader.key)["ready"] == 3)
        assert reader[1]["raw_prompts"] == "cc"
        assert pool.get_stats()["submitted"] == 3
    finally:
        release.set()
        reader.close()
        pool.close()


def test_pool_restart_reconstructs_consumed_prefix_without_requery(tmp_path):
    pool = TeacherPool(Teacher(), workers=2)
    first = dataset(tmp_path, _LocalPoolRequestClient(pool), lookahead=1)
    try:
        assert [first[i]["raw_prompts"] for i in range(5)] == ["aa", "cc", "ee", "ff", "gg"]
    finally:
        first.close()
        pool.close()
    pool = TeacherPool(Teacher(), workers=2)
    resumed = dataset(tmp_path, _LocalPoolRequestClient(pool), lookahead=1, start_index=4)
    try:
        assert resumed[4]["raw_prompts"] == "gg"
        wait_for(lambda: pool.get_stats()["submitted"] == 2)
        assert pool.get_stats()["submitted"] == 2  # current + one future, no consumed prefix
        assert resumed.rejected_count() == 2
    finally:
        resumed.close()
        pool.close()


def test_service_error_stays_visible_and_does_not_replace_samples(pool_port, tmp_path):
    client = RemoteTeacherPoolClient(Teacher(), port=pool_port)
    reader = dataset(tmp_path, client, lookahead=1, prompts=["zz", "aa"])
    try:
        with pytest.raises(RuntimeError, match="service offline"):
            reader[0]
        assert reader.rejected_count() == 0
    finally:
        reader.close()


def test_evicted_results_can_be_replayed_without_changing_selection(pool_port, tmp_path):
    reader = dataset(tmp_path, RemoteTeacherPoolClient(Teacher(), port=pool_port), lookahead=1)
    try:
        before = copy.deepcopy(reader[0])
        assert reader[6]["raw_prompts"] == "ii"
        assert reader.teacher.sample_stream_stats(reader.key)["evicted"] > 0
        replayed = reader[0]
        for key in before:
            if isinstance(before[key], torch.Tensor):
                assert torch.equal(before[key], replayed[key])
            else:
                assert before[key] == replayed[key]
    finally:
        reader.close()


def test_changed_source_is_rejected_by_pool_even_with_original_signature(pool_port, tmp_path):
    client = RemoteTeacherPoolClient(Teacher(), port=pool_port)
    source, signature = source_spec(tmp_path)
    (tmp_path / "train.jsonl").write_text('{"text":"aa"}\n')
    with pytest.raises(ValueError, match="source/tokenizer configuration mismatch"):
        PoolValidatedDataset(source, client, tmp_path / "selection", signature)


def test_tiny_capacity_preserves_concurrent_readers_and_global_worker_bound(tmp_path):
    pool = TeacherPool(Teacher(), workers=1, max_results=1, queue_size=1)
    reader = dataset(tmp_path, _LocalPoolRequestClient(pool), lookahead=8, count=6)
    try:
        def read(_):
            return [reader[i]['raw_prompts'] for i in range(6)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read, i) for i in range(4)]
            result = [future.result(timeout=15) for future in futures]
        assert all(rows == ['aa', 'cc', 'ee', 'ff', 'gg', 'hh'] for rows in result)
        stats = pool.get_stats()
        assert stats['peak_active'] == 1
        assert reader.stats()['ready'] <= reader.stats()['capacity'] == 1
        assert reader.rejected_count() == 2
    finally:
        reader.close()
        pool.close()


def test_validation_stream_does_not_compete_for_training_requests_until_consumed(tmp_path):
    entered = threading.Event()

    class Signaling(Teacher):
        def get_logits(self, *args, **kwargs):
            entered.set()
            return super().get_logits(*args, **kwargs)

    pool = TeacherPool(Signaling(), workers=2)
    source, signature = source_spec(tmp_path, prompts=['aa', 'cc', 'ee'], count=3)
    settings = json.loads(signature)
    settings['split'] = 'validation'
    reader = PoolValidatedDataset(source, _LocalPoolRequestClient(pool),
                                  tmp_path / 'selection', json.dumps(settings, sort_keys=True),
                                  lookahead=2)
    try:
        assert not entered.wait(0.15), 'Validation must not prefill the HTTP queue during training'
        assert pool.get_stats()['submitted'] == 0
        assert reader[0]['raw_prompts'] == 'aa'
        assert entered.is_set()
        wait_for(lambda: reader.stats()['ready'] == 3)
    finally:
        reader.close()
        pool.close()


def test_step_deadline_bounds_a_pending_validated_sample(tmp_path):
    release = threading.Event()

    class Blocking(Teacher):
        def get_logits(self, *args, **kwargs):
            assert release.wait(10)
            return super().get_logits(*args, **kwargs)

    pool = TeacherPool(Blocking(), workers=1)
    client = _LocalPoolRequestClient(pool)
    reader = dataset(tmp_path, client, lookahead=1, prompts=['aa'], count=1)
    try:
        with teacher_request_deadline(client, time.monotonic() + 0.02):
            with pytest.raises(TimeoutError, match='validated-sample queue timed out'):
                reader[0]
        assert client._thread_local.deadline is None
    finally:
        release.set()
        reader.close()
        pool.close()
