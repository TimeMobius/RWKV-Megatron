"""Protect partial PP deliveries, ACK after receipt, and isolate DP ownership."""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from rwkv_megatron.teacher_pool import RemoteTeacherPoolClient
from rwkv_megatron.teacher_sample_stream import PoolValidatedDataset, ValidatedSampleStream
from test_teacher_sample_stream import Teacher, pool_port, source_spec, wait_for


class Samples:
    def __init__(self, count=8):
        self.count = count
        self.calls = Counter()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.calls[index] += 1
        return dict(tokens=torch.tensor([index]))

    def rejected_count(self):
        return 0

    def close(self):
        pass


def layout(dp=1, cp=1, micro=1, pp=(0, 1), offset=0):
    return dict(dp_size=dp, cp_size=cp, micro_batch_size=micro,
                pp_stages=list(pp), offset=offset)


def ack(stream, consumer, index, result):
    stream.acknowledge(consumer, index, result['receipt'])


def test_partial_sample_survives_full_cache_and_late_pipeline_registration():
    source = Samples()
    stream = ValidatedSampleStream(source, 2, 2, delivery=layout())
    stream.register_reader('first', (0, 0, 0))
    try:
        row = stream.get(0, 1, consumer='first')
        ack(stream, 'first', 0, row)
        wait_for(lambda: stream.stats()['ready'] == 2)
        assert stream.sample_status(0)['state'] == 'partial'
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(stream.get, 2, 1, 'first')
            wait_for(lambda: 2 in stream._wanted)
            assert not future.done()
            assert stream.sample_status(0)['state'] == 'partial'
            assert source.calls[2] == 0, 'Do not request past a full protected cache'
            # Expected readers are known before they register; PP0 cannot
            # evict this entry simply because PP1 has not started yet.
            stream.register_reader('last', (0, 1, 0))
            second = stream.get(0, 1, consumer='last')
            assert second['receipt'] == row['receipt']
            ack(stream, 'last', 0, second)
            assert future.result(timeout=2)['sample']['tokens'][1] == (1,)
        assert stream.sample_status(0)['state'] == 'absent'
        assert source.calls[0] == 1, 'A slow PP reader must reuse the original result'
        assert stream.stats()['ready'] == 2
    finally:
        stream.close()
        stream.join()


def test_dp_samples_only_wait_for_their_owner_pipeline_endpoints():
    stream = ValidatedSampleStream(Samples(4), 4, 4, delivery=layout(dp=2, micro=2))
    stream.register_reader('dp0', (0, 0, 0))
    stream.register_reader('dp1-first', (1, 0, 0))
    stream.register_reader('dp1-last', (1, 1, 0))
    try:
        with pytest.raises(ValueError, match='not assigned'):
            stream.get(2, consumer='dp0')
        first = stream.get(2, 1, consumer='dp1-first')
        ack(stream, 'dp1-first', 2, first)
        status = stream.sample_status(2)
        assert status == dict(state='partial', required=[(1, 0, 0), (1, 1, 0)],
                              received=[(1, 0, 0)])
        second = stream.get(2, 1, consumer='dp1-last')
        ack(stream, 'dp1-last', 2, second)
        assert stream.sample_status(2)['state'] == 'received'
    finally:
        stream.close()
        stream.join()


def test_all_context_and_pipeline_endpoints_must_confirm_their_copy():
    stream = ValidatedSampleStream(Samples(1), 1, 1, delivery=layout(cp=2))
    readers = [(0, pp, cp) for pp in (0, 1) for cp in (0, 1)]
    try:
        for index, reader in enumerate(readers):
            name = str(reader)
            stream.register_reader(name, reader)
            result = stream.get(0, 1, consumer=name)
            assert stream.sample_status(0)['state'] != 'received'
            ack(stream, name, 0, result)
            assert stream.sample_status(0)['state'] == ('received' if index == 3 else 'partial')
        assert stream.dataset.calls[0] == 1
    finally:
        stream.close()
        stream.join()


def test_ack_is_idempotent_and_old_generation_cannot_ack_replayed_result():
    stream = ValidatedSampleStream(Samples(2), 1, 1, delivery=layout(pp=(0,)))
    stream.register_reader('only', (0, 0, 0))
    try:
        old = stream.get(0, 1, consumer='only')
        ack(stream, 'only', 0, old)
        ack(stream, 'only', 0, old)
        second = stream.get(1, 1, consumer='only')
        ack(stream, 'only', 1, second)
        replay = stream.get(0, 1, consumer='only')
        assert old['receipt'] != replay['receipt']
        ack(stream, 'only', 0, old)
        assert stream.sample_status(0)['state'] == 'unclaimed'
        ack(stream, 'only', 0, replay)
        assert stream.sample_status(0)['state'] == 'received'
    finally:
        stream.close()
        stream.join()


def test_disconnecting_reader_does_not_mark_partial_data_received():
    stream = ValidatedSampleStream(Samples(1), 1, 1, delivery=layout())
    stream.register_reader('first', (0, 0, 0))
    stream.register_reader('last', (0, 1, 0))
    try:
        row = stream.get(0, 1, consumer='first')
        ack(stream, 'first', 0, row)
        stream.unregister_reader('last')
        assert stream.sample_status(0)['state'] == 'partial'
        assert stream._eviction_candidate() is None
    finally:
        stream.close()
        stream.join()


def test_cache_rejects_capacity_too_small_to_assemble_a_dp_microbatch():
    with pytest.raises(ValueError, match='complete DP microbatch'):
        ValidatedSampleStream(Samples(), 1, 1, delivery=layout(micro=2))


def test_separate_pool_marks_receipt_only_after_client_unpacks(pool_port, tmp_path, monkeypatch):
    import rwkv_megatron.teacher_sample_stream as module

    source, signature = source_spec(tmp_path, prompts=['aa', 'cc', 'ee'], count=3)
    client = RemoteTeacherPoolClient(Teacher(), port=pool_port)
    first = PoolValidatedDataset(source, client, tmp_path/'selection', signature,
                                 lookahead=1, delivery=layout(), reader=(0, 0, 0))
    last = PoolValidatedDataset(source, client, tmp_path/'selection', signature,
                                lookahead=1, delivery=layout(), reader=(0, 1, 0))
    original = module.unpack_sample
    try:
        def broken(_):
            raise ValueError('injected decode failure')
        monkeypatch.setattr(module, 'unpack_sample', broken)
        with pytest.raises(ValueError, match='decode failure'):
            first[0]
        assert client.sample_delivery_status(first.key, 0)['state'] == 'unclaimed'
        monkeypatch.setattr(module, 'unpack_sample', original)
        assert first[0]['raw_prompts'] == 'aa'
        assert client.sample_delivery_status(first.key, 0)['state'] == 'partial'
        assert last[0]['raw_prompts'] == 'aa'
        assert client.sample_delivery_status(first.key, 0)['state'] == 'received'
        assert first.stats()['receipt_tracking'] == 1
        assert first.stats()['partial'] == 0
    finally:
        monkeypatch.setattr(module, 'unpack_sample', original)
        first.close()
        last.close()


def test_megatron_context_matches_sampler_and_registers_only_endpoint_sources(monkeypatch):
    from megatron.core import parallel_state as ps
    import megatron.training
    from rwkv_megatron.teacher_sample_stream import megatron_delivery_context

    args = SimpleNamespace(dataloader_type='single', micro_batch_size=2)
    monkeypatch.setattr(megatron.training, 'get_args', lambda: args)
    for name, value in dict(is_initialized=True, get_pipeline_model_parallel_rank=1,
                            get_pipeline_model_parallel_world_size=2,
                            get_tensor_model_parallel_rank=0, get_data_parallel_world_size=2,
                            get_context_parallel_world_size=2, get_data_parallel_rank=1,
                            get_context_parallel_rank=0).items():
        monkeypatch.setattr(ps, name, lambda value=value: value)
    delivery, reader = megatron_delivery_context(16)
    assert delivery == layout(dp=2, cp=2, micro=2, offset=16)
    assert reader == (1, 1, 0)
    args.dataloader_type = 'cyclic'
    with pytest.raises(ValueError, match='single dataloader'):
        megatron_delivery_context(16)


def test_dp2_pp2_cp2_readers_progress_under_cache_pressure_without_requery(pool_port, tmp_path):
    from megatron.training.datasets.data_samplers import MegatronPretrainingSampler

    source, signature = source_spec(tmp_path, count=12)
    client = RemoteTeacherPoolClient(Teacher(), port=pool_port)
    readers = []
    try:
        for dp in range(2):
            for pp in range(2):
                for cp in range(2):
                    reader = PoolValidatedDataset(
                        source, client, tmp_path/'selection', signature, lookahead=2,
                        delivery=layout(dp=2, cp=2, micro=2), reader=(dp, pp, cp),
                    )
                    readers.append(((dp, pp, cp), reader))

        def consume(item):
            identity, reader = item
            sampler = MegatronPretrainingSampler(
                total_samples=12, consumed_samples=0, micro_batch_size=2,
                data_parallel_rank=identity[0], data_parallel_size=2,
            )
            actual = []
            for batch in sampler:
                for index in batch:
                    row = reader[index]
                    actual.append((index, row['raw_prompts']))
                    assert torch.equal(row['teacher_token_ids'][:, 0], row['tokens'])
            return identity, actual

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(consume, item) for item in readers]
            results = [future.result(timeout=20) for future in futures]
        expected = ['aa', 'cc', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj', 'kk', 'll', 'mm', 'nn']
        for identity, actual in results:
            indices = [index for index in range(12) if index // 2 % 2 == identity[0]]
            assert actual == [(index, expected[index]) for index in indices]
        assert client.get_stats()['pool_submitted'] == 14  # 12 good + 2 bad, shared by 8 readers
        stats = readers[0][1].stats()
        assert stats['receipt_tracking'] == 1
        assert stats['ready'] <= stats['capacity'] == 4
        assert stats['partial'] == 0
    finally:
        for _, reader in readers:
            reader.close()


def test_repeated_validation_uses_fresh_receipts_for_each_pipeline_read_round(pool_port, tmp_path):
    source, signature = source_spec(tmp_path, prompts=['aa', 'cc', 'ee'], count=3)
    client = RemoteTeacherPoolClient(Teacher(), port=pool_port)
    readers = [PoolValidatedDataset(source, client, tmp_path/'selection', signature,
                                    lookahead=1, delivery=layout(), reader=(0, pp, 0))
               for pp in range(2)]
    try:
        for round_index in range(3):
            def consume(reader):
                return [reader[index]['raw_prompts'] for index in range(3)]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(consume, reader) for reader in readers]
                assert [future.result(timeout=10) for future in futures] == [['aa', 'cc', 'ee']] * 2
            assert readers[0].key == readers[1].key
            assert readers[0]._spec['delivery_epoch'] == round_index
            assert readers[0].stats()['partial'] == 0
        assert client.get_stats()['pool_sample_streams'] == 1
        assert client.get_stats()['pool_submitted'] == 9
    finally:
        for reader in readers:
            reader.close()


def test_non_reader_ranks_skip_jsonl_registration(monkeypatch):
    import pretrain_rwkv7 as pretrain

    args = SimpleNamespace(data_path=['fixture.jsonl'], rwkv_distill=True,
                           rwkv_teacher_replace_bad_samples=True)
    monkeypatch.setattr(pretrain, 'get_args', lambda: args)
    monkeypatch.setattr(pretrain, 'is_dataset_built_on_rank', lambda: False)
    monkeypatch.setattr(pretrain, '_get_online_teacher',
                        lambda _: pytest.fail('Non-readers must not register with the pool'))
    assert pretrain.train_valid_test_datasets_provider([8, 8, 8]) == (None, None, None)
