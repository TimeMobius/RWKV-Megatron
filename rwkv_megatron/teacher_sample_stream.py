"""Pool-owned production of validated input/target pairs, with bounded lookahead."""

import hashlib
import json
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def pack_sample(sample):
    """Avoid torch multiprocessing storage handles across the manager RPC."""
    return {
        key: (value.numpy().dtype.str, tuple(value.shape), value.numpy().tobytes())
        if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }


def unpack_sample(sample):
    return {
        key: torch.from_numpy(np.frombuffer(value[2], dtype=value[0]).reshape(value[1]).copy())
        if isinstance(value, tuple) else value
        for key, value in sample.items()
    }


def megatron_delivery_context(start_index):
    """Match the single dataloader's DP slices; only TP source ranks read data."""
    parallel = sys.modules.get("megatron.core.parallel_state")
    if parallel is None or not parallel.is_initialized():
        return None, None
    from megatron.training import get_args

    args = get_args()
    if args.dataloader_type != "single":
        raise ValueError("Teacher receipt tracking requires the single dataloader")
    pp = parallel.get_pipeline_model_parallel_rank()
    last = parallel.get_pipeline_model_parallel_world_size() - 1
    if parallel.get_tensor_model_parallel_rank() != 0 or pp not in (0, last):
        raise ValueError("Only pipeline endpoint TP source ranks may read teacher samples")
    delivery = dict(dp_size=parallel.get_data_parallel_world_size(),
                    cp_size=parallel.get_context_parallel_world_size(),
                    pp_stages=sorted({0, last}), micro_batch_size=args.micro_batch_size,
                    offset=start_index)
    reader = (parallel.get_data_parallel_rank(), pp, parallel.get_context_parallel_rank())
    return delivery, reader


class ValidatedSampleStream:
    """One ordered selector per corpus, shared by all DP/PP readers.

    The producer is independent of GPU progress. Ready pairs are reusable by
    several readers; eviction only drops disposable tensors, never selection
    decisions. With a delivery layout, entries are protected until every required
    DP/PP/CP reader acknowledges its own copy. Unconfigured standalone clients
    retain the older disposable LRU cache behavior.
    """

    def __init__(self, dataset, lookahead, capacity, start_index=0, eager=True,
                 delivery=None):
        self.dataset = dataset
        self.lookahead = min(lookahead, capacity)
        self.capacity = capacity
        self._condition = threading.Condition()
        self._ready = OrderedDict()
        self._wanted = OrderedDict()
        self.delivery = delivery
        self._readers = {}
        self._received = {}
        self._versions = {}
        if delivery is not None:
            for key in ("dp_size", "cp_size", "micro_batch_size"):
                if type(delivery.get(key)) is not int or delivery[key] < 1:
                    raise ValueError(f"Invalid delivery {key}")
            stages = delivery.get("pp_stages")
            if (not isinstance(stages, list) or not stages
                    or any(type(stage) is not int or stage < 0 for stage in stages)
                    or len(set(stages)) != len(stages)
                    or type(delivery.get("offset")) is not int or delivery["offset"] < 0):
                raise ValueError("Invalid delivery stages/offset")
            if capacity < delivery["micro_batch_size"] * delivery["dp_size"]:
                raise ValueError("Teacher cache must hold one complete DP microbatch")
        self._high_water = start_index - 1
        self._cursor = start_index
        self._activated = eager
        self._error = None
        self._closed = False
        self._stats = dict(produced=0, cache_hits=0, cache_misses=0, evicted=0,
                           served=0, queue_wait_seconds=0.0)
        self._thread = threading.Thread(target=self._produce, daemon=True,
                                        name="teacher-sample-producer")
        self._thread.start()

    def register_reader(self, consumer, reader):
        if self.delivery is None:
            return
        reader = tuple(reader)
        if (len(reader) != 3 or any(type(value) is not int for value in reader)
                or not 0 <= reader[0] < self.delivery["dp_size"]
                or reader[1] not in self.delivery["pp_stages"]
                or not 0 <= reader[2] < self.delivery["cp_size"]):
            raise ValueError("Invalid teacher reader DP/PP/CP identity")
        with self._condition:
            if reader in self._readers.values():
                raise ValueError("Teacher reader identity is already registered")
            self._readers[consumer] = reader

    def unregister_reader(self, consumer):
        with self._condition:
            self._readers.pop(consumer, None)
            # Disconnecting is not a delivery acknowledgement. Partial results
            # remain protected; distributed failure handling owns lost peers.
            self._condition.notify_all()

    def _required(self, index):
        layout = self.delivery
        owner = ((index - layout["offset"]) // layout["micro_batch_size"]) % layout["dp_size"]
        return {(owner, pp, cp) for pp in layout["pp_stages"]
                for cp in range(layout["cp_size"])}

    def _reader(self, consumer, index):
        reader = self._readers.get(consumer)
        if reader is None or reader not in self._required(index):
            raise ValueError("Teacher sample is not assigned to this DP/PP/CP reader")
        return reader

    def _reclaimable(self, index):
        return (index not in self._wanted and
                (self.delivery is None or self._required(index) <= self._received[index]))

    def _eviction_candidate(self):
        return next((index for index in self._ready if self._reclaimable(index)), None)

    def acknowledge(self, consumer, index, receipt):
        with self._condition:
            if self.delivery is None:
                return
            reader = self._reader(consumer, index)
            # Ignore duplicate/delayed ACKs for a previous cached generation.
            if self._versions.get(index) != receipt:
                return
            self._received[index].add(reader)
            self._condition.notify_all()

    def sample_status(self, index):
        with self._condition:
            if self.delivery is None:
                return dict(state="untracked")
            required = self._required(index)
            received = self._received.get(index, set())
            state = ("absent" if index not in self._ready else
                     "received" if required <= received else
                     "partial" if received else "unclaimed")
            return dict(state=state, required=sorted(required), received=sorted(received))

    def get(self, index, timeout=1.0, consumer=None):
        if type(index) is not int or not 0 <= index < len(self.dataset):
            raise IndexError(index)
        with self._condition:
            if self._closed:
                raise RuntimeError("Teacher sample stream is closed")
            if self.delivery is not None:
                self._reader(consumer, index)
            self._high_water = max(self._high_water, index)
            self._activated = True
            self._wanted[index] = self._wanted.get(index, 0) + 1
            self._condition.notify_all()
            ready_on_entry = index in self._ready
            wait_start = time.monotonic()
            try:
                self._condition.wait_for(
                    lambda: index in self._ready or self._error is not None or self._closed,
                    timeout=min(max(timeout, 0.0), 1.0),
                )
                self._stats["queue_wait_seconds"] += time.monotonic() - wait_start
                self._stats["cache_hits" if ready_on_entry else "cache_misses"] += 1
                if index in self._ready:
                    self._ready.move_to_end(index)
                    self._stats["served"] += 1
                    return dict(ok=True, sample=self._ready[index],
                                receipt=self._versions.get(index))
                if self._error is not None:
                    return dict(ok=False, error=self._error)
                if self._closed:
                    raise RuntimeError("Teacher sample stream is closed")
                return None
            finally:
                waiting = self._wanted.get(index, 0) - 1
                if waiting > 0:
                    self._wanted[index] = waiting
                else:
                    self._wanted.pop(index, None)
                self._condition.notify_all()

    def _produce(self):
        try:
            while True:
                with self._condition:
                    while True:
                        if self._closed:
                            return
                        if not self._activated:
                            self._condition.wait()
                            continue
                        if (len(self._ready) >= self.capacity
                                and self._eviction_candidate() is None):
                            self._condition.wait()
                            continue
                        missing = next((i for i in self._wanted if i not in self._ready), None)
                        if missing is not None:
                            # Fill the unconsumed prefix once even if DP1 asks
                            # before DP0; later PP readers reuse those results.
                            index = min(missing, self._cursor)
                            if index == self._cursor:
                                self._cursor += 1
                            break
                        limit = min(len(self.dataset), self._high_water + 1 + self.lookahead)
                        if self._cursor < limit:
                            index = self._cursor
                            self._cursor += 1
                            if index in self._ready:
                                continue
                            break
                        self._condition.wait()
                # All dataset/tokenizer/validation state belongs to this thread
                # and the selector's HTTP workers, never a student thread.
                sample = pack_sample(self.dataset[index])
                with self._condition:
                    if self._closed:
                        return
                    while len(self._ready) >= self.capacity:
                        evict = self._eviction_candidate()
                        if evict is not None:
                            del self._ready[evict]
                            self._received.pop(evict, None)
                            self._versions.pop(evict, None)
                            self._stats["evicted"] += 1
                            break
                        if self._closed:
                            return
                        self._condition.notify_all()
                        self._condition.wait()
                    self._ready[index] = sample
                    if self.delivery is not None:
                        self._received[index] = set()
                        self._versions[index] = uuid.uuid4().hex
                    self._stats["produced"] += 1
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = f"{type(error).__name__}: {error}"
                self._condition.notify_all()
        finally:
            self.dataset.close()

    def stats(self):
        with self._condition:
            stats = dict(self._stats, ready=len(self._ready), capacity=self.capacity,
                         failed=int(self._error is not None),
                         receipt_tracking=int(self.delivery is not None))
            states = [self.sample_status(index)["state"] for index in self._ready] if self.delivery else []
            stats.update(unclaimed=states.count("unclaimed"), partial=states.count("partial"),
                         received=states.count("received"))
        stats["rejected_source_samples"] = self.dataset.rejected_count()
        return stats

    def close(self):
        with self._condition:
            self._closed = True
            self._ready.clear()
            self._wanted.clear()
            self._received.clear()
            self._versions.clear()
            self._condition.notify_all()

    def join(self):
        self._thread.join()

    def is_alive(self):
        return self._thread.is_alive()


class PoolValidatedDataset(Dataset):
    """Student-side RPC reader; no teacher requests or selection state here."""

    pool_validated = True

    def __init__(self, source, teacher, directory, signature, lookahead=4, start_index=0,
                 delivery=None, reader=None):
        if delivery is None and reader is None:
            delivery, reader = megatron_delivery_context(start_index)
        spec = dict(paths=[str(path.resolve()) for path in source._paths],
                    sequence_length=source.sequence_length, target_samples=len(source),
                    text_field=source.text_field, chunk_chars=source.chunk_chars,
                    max_records=json.loads(signature)["max_records"],
                    directory=str(Path(directory).resolve()),
                    signature=signature, lookahead=lookahead, start_index=start_index)
        if delivery is not None:
            spec["delivery"] = delivery
            spec["delivery_epoch"] = 0
        self._initialize(teacher, spec, reader)

    def _initialize(self, teacher, spec, reader=None):
        self.teacher, self.count = teacher, spec["target_samples"]
        self._spec, self._reader_identity, self._last_index = spec, reader, None
        # Including the directory prevents unrelated runs sharing tensor caches.
        self.key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        self.consumer = teacher.open_sample_stream(self.key, spec, reader)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.consumer is None:
            raise RuntimeError("Teacher sample reader is closed")
        index = int(index)
        if not 0 <= index < self.count:
            raise IndexError(index)
        if ("delivery" in self._spec and self._last_index is not None
                and index <= self._last_index):
            # Fixed validation restarts its sampler. Each endpoint starts the
            # same new round independently; old ACKs cannot release new data.
            spec = dict(self._spec, delivery_epoch=self._spec["delivery_epoch"] + 1,
                        start_index=0, delivery=dict(self._spec["delivery"], offset=0))
            key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            consumer = self.teacher.open_sample_stream(key, spec, self._reader_identity)
            previous_key, previous_consumer = self.key, self.consumer
            self._spec, self.key, self.consumer = spec, key, consumer
            self.teacher.close_sample_stream(previous_key, previous_consumer)
        sample = self.teacher.get_validated_sample(self.key, self.consumer, index)
        self._last_index = index
        return sample

    def rejected_count(self):
        return self.stats()["rejected_source_samples"]

    def stats(self):
        return self.teacher.sample_stream_stats(self.key)

    def close(self):
        if self.consumer is not None:
            self.teacher.close_sample_stream(self.key, self.consumer)
            self.consumer = None
