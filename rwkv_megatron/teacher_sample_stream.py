"""Pool-owned production of validated input/target pairs, with bounded lookahead."""

import hashlib
import json
import threading
import time
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


class ValidatedSampleStream:
    """One ordered selector per corpus, shared by all DP/PP readers.

    The producer is independent of GPU progress. Ready pairs are reusable by
    several readers; eviction only drops disposable tensors, never selection
    decisions. A lagging reader can regenerate an evicted pair from the journal.
    """

    def __init__(self, dataset, lookahead, capacity, start_index=0):
        self.dataset = dataset
        self.lookahead = min(lookahead, capacity)
        self.capacity = capacity
        self._condition = threading.Condition()
        self._ready = OrderedDict()
        self._wanted = OrderedDict()
        self._high_water = start_index - 1
        self._cursor = start_index
        self._error = None
        self._closed = False
        self._stats = dict(produced=0, cache_hits=0, cache_misses=0, evicted=0,
                           served=0, queue_wait_seconds=0.0)
        self._thread = threading.Thread(target=self._produce, daemon=True,
                                        name="teacher-sample-producer")
        self._thread.start()

    def get(self, index, timeout=1.0):
        if type(index) is not int or not 0 <= index < len(self.dataset):
            raise IndexError(index)
        with self._condition:
            if self._closed:
                raise RuntimeError("Teacher sample stream is closed")
            self._high_water = max(self._high_water, index)
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
                    return dict(ok=True, sample=self._ready[index])
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
                        if (len(self._ready) >= self.capacity
                                and all(i in self._wanted for i in self._ready)):
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
                        evict = next((i for i in self._ready if i not in self._wanted), None)
                        if evict is not None:
                            del self._ready[evict]
                            self._stats["evicted"] += 1
                            break
                        if self._closed:
                            return
                        self._condition.notify_all()
                        self._condition.wait()
                    self._ready[index] = sample
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
                         failed=int(self._error is not None))
        stats["rejected_source_samples"] = self.dataset.rejected_count()
        return stats

    def close(self):
        with self._condition:
            self._closed = True
            self._ready.clear()
            self._wanted.clear()
            self._condition.notify_all()

    def join(self):
        self._thread.join()


class PoolValidatedDataset(Dataset):
    """Student-side RPC reader; no teacher requests or selection state here."""

    pool_validated = True

    def __init__(self, source, teacher, directory, signature, lookahead=4, start_index=0):
        spec = dict(paths=[str(path.resolve()) for path in source._paths],
                    sequence_length=source.sequence_length, target_samples=len(source),
                    text_field=source.text_field, chunk_chars=source.chunk_chars,
                    max_records=json.loads(signature)["max_records"],
                    directory=str(Path(directory).resolve()),
                    signature=signature, lookahead=lookahead, start_index=start_index)
        self._initialize(teacher, spec)

    def _initialize(self, teacher, spec):
        self.teacher, self.count = teacher, spec["target_samples"]
        # Including the directory prevents unrelated runs sharing tensor caches.
        self.key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        self.consumer = teacher.open_sample_stream(self.key, spec)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.consumer is None:
            raise RuntimeError("Teacher sample reader is closed")
        return self.teacher.get_validated_sample(self.key, self.consumer, int(index))

    def rejected_count(self):
        return self.stats()["rejected_source_samples"]

    def stats(self):
        return self.teacher.sample_stream_stats(self.key)

    def close(self):
        if self.consumer is not None:
            self.teacher.close_sample_stream(self.key, self.consumer)
            self.consumer = None
