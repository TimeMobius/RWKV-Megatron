"""Node-local teacher RPC pool with bounded, disposable CPU results.

BaseManager is intended for trusted local processes. No CUDA/Megatron state or
training cursor lives here; requests are namespaced UUIDs, not iteration IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections import deque
from multiprocessing.managers import BaseManager

import numpy as np
import torch

from rwkv_megatron.online_teacher import TeacherTopK


def teacher_metadata(client):
    tokenizer = client._get_tokenizer()
    vocabulary = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=True).encode()
    return dict(protocol=1, server_url=client.server_url, model=client.model_name,
                top_k=client.top_k, seed=client.request_seed,
                request_temperature=client.request_temperature,
                prompt_mode=getattr(client, 'prompt_mode', 'text'),
                tokenizer_use_fast=getattr(client, 'tokenizer_use_fast', True),
                valid_vocab_size=client.get_effective_vocab_size(),
                vocab_sha256=hashlib.sha256(vocabulary).hexdigest())


class TeacherPool:
    """Shared queue, adaptive worker limit, and consumable in-memory results.

    Capacity is reserved on submission, so unread results are never silently
    evicted to make room for new work. Finished orphan jobs expire after TTL.
    """

    def __init__(self, client, workers=16, queue_size=256, max_results=512,
                 result_ttl=300.0, min_workers=1, latency_target=2.0):
        if any(type(n) is not int or n < 1 for n in (workers, queue_size, max_results, min_workers)):
            raise ValueError("pool capacities must be positive integers")
        if min_workers > workers or any(not math.isfinite(n) or n <= 0 for n in (result_ttl, latency_target)):
            raise ValueError("invalid pool concurrency/TTL settings")
        self.client = client
        self.metadata = teacher_metadata(client)
        self.workers, self.queue_size, self.max_results = workers, queue_size, max_results
        self.result_ttl, self.min_workers, self.latency_target = result_ttl, min_workers, latency_target
        self._cond = threading.Condition()
        self._queue = deque()
        self._jobs = {}
        self._closed = False
        self._active = 0
        self._target = workers
        self._stats = dict(submitted=0, completed=0, failed=0, consumed=0, cancelled=0,
                           expired=0, rejected=0, peak_active=0, concurrency_reductions=0,
                           concurrency_increases=0)
        self._threads = [threading.Thread(target=self._worker, daemon=True, name=f"teacher-pool-{i}")
                         for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def get_metadata(self):
        return dict(self.metadata)

    def _expire(self):
        now = time.monotonic()
        for key, job in list(self._jobs.items()):
            if job["finished"] is not None and now - job["finished"] >= self.result_ttl:
                del self._jobs[key]
                self._stats["expired"] += 1

    def submit(self, request_id, ids_bytes, mask_bytes, shape, prompts):
        # Explicit arrays avoid torch's shared-memory/CUDA pickling machinery.
        if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("input shape must be [batch, sequence]")
        count = math.prod(shape)
        if len(ids_bytes) != count * 8 or (mask_bytes is not None and len(mask_bytes) != count):
            raise ValueError("invalid input tensor byte lengths")
        fingerprint = hashlib.sha256(ids_bytes + (mask_bytes or b"") +
                                     json.dumps([shape, prompts], ensure_ascii=True).encode()).digest()
        with self._cond:
            if self._closed:
                raise RuntimeError("TeacherPool is closed")
            self._expire()
            if request_id in self._jobs:
                if self._jobs[request_id]["fingerprint"] != fingerprint:
                    raise ValueError("request ID reused for different input")
                return True
            if len(self._queue) >= self.queue_size or len(self._jobs) >= self.max_results:
                self._stats["rejected"] += 1
                return False
            self._jobs[request_id] = dict(fingerprint=fingerprint, finished=None, result=None)
            self._queue.append((request_id, ids_bytes, mask_bytes, tuple(shape), prompts))
            self._stats["submitted"] += 1
            self._cond.notify_all()
            return True

    def get(self, request_id, timeout=1.0):
        deadline = time.monotonic() + min(max(float(timeout), 0.0), 5.0)
        with self._cond:
            while True:
                self._expire()
                if request_id not in self._jobs:
                    raise RuntimeError("teacher pool request expired, cancelled, or unknown")
                job = self._jobs[request_id]
                if job["finished"] is not None:
                    del self._jobs[request_id]
                    self._stats["consumed"] += 1
                    return job["result"]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def cancel(self, request_id):
        with self._cond:
            if self._jobs.pop(request_id, None) is not None:
                self._queue = deque(item for item in self._queue if item[0] != request_id)
                self._stats["cancelled"] += 1
                self._cond.notify_all()

    def _worker(self):
        while True:
            with self._cond:
                while not self._closed and (not self._queue or self._active >= self._target):
                    self._expire()
                    self._cond.wait(1.0)
                if self._closed:
                    return
                request_id, ids_bytes, mask_bytes, shape, prompts = self._queue.popleft()
                self._active += 1
                self._stats["peak_active"] = max(self._stats["peak_active"], self._active)
            try:
                tokens = torch.from_numpy(np.frombuffer(ids_bytes, dtype=np.int64).reshape(shape).copy())
                mask = (None if mask_bytes is None else
                        torch.from_numpy(np.frombuffer(mask_bytes, dtype=np.bool_).reshape(shape).copy()))
                out = self.client.get_logits(tokens, attention_mask=mask, prompts=prompts)
                metrics = self.client.get_last_request_metrics()
                result = dict(ok=True, shape=tuple(out.token_ids.shape), metrics=metrics,
                              token_ids=out.token_ids.cpu().numpy().astype(np.int32).tobytes(),
                              logprobs=out.logprobs.cpu().numpy().astype(np.float32).tobytes())
            except Exception as error:
                result = dict(ok=False, error=f"{type(error).__name__}: {error}")
            with self._cond:
                self._active -= 1
                self._stats["completed" if result["ok"] else "failed"] += 1
                if result["ok"]:
                    metrics = result["metrics"]
                    latency = metrics.get("latency_seconds", 0.0)
                    old = self._target
                    if metrics.get("rate_limits", 0):
                        self._target = max(self.min_workers, old // 2)
                    elif latency > self.latency_target:
                        self._target = max(self.min_workers, old - 1)
                    elif latency and latency < self.latency_target * 0.5:
                        self._target = min(self.workers, old + 1)
                    if old != self._target:
                        key = "concurrency_reductions" if self._target < old else "concurrency_increases"
                        self._stats[key] += 1
                if request_id in self._jobs:
                    self._jobs[request_id].update(result=result, finished=time.monotonic())
                self._cond.notify_all()

    def get_stats(self):
        with self._cond:
            self._expire()
            stats = dict(self._stats, active=self._active, queued=len(self._queue),
                         retained=len(self._jobs), ready=sum(j["finished"] is not None for j in self._jobs.values()),
                         target_workers=self._target)
        stats.update({f"client_{k}": v for k, v in self.client.get_stats().items()})
        return stats

    def close(self):
        with self._cond:
            self._closed = True
            self._queue.clear()
            self._jobs.clear()
            self._cond.notify_all()
        for thread in self._threads:
            thread.join()


def make_pool_server(pool, host="127.0.0.1", port=50055, authkey="teacher-pool"):
    class Manager(BaseManager):
        pass

    Manager.register("get_pool", callable=lambda: pool,
                     exposed=("get_metadata", "submit", "get", "cancel", "get_stats"))
    return Manager(address=(host, port), authkey=authkey.encode()).get_server()


class RemoteTeacherPoolClient:
    """Drop-in teacher client: no direct-vLLM fallback or local result cache."""

    def __init__(self, local_client, host="127.0.0.1", port=50055, authkey="teacher-pool", timeout=2700.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("pool timeout must be finite and positive")
        self.local_client = local_client
        self.prompt_mode = getattr(local_client, 'prompt_mode', 'text')
        self.timeout = timeout
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._stats = dict(requests=0, backpressure_waits=0)

        class Manager(BaseManager):
            pass

        Manager.register("get_pool")
        self._manager = Manager(address=(host, port), authkey=authkey.encode())
        self._manager.connect()
        self._pool = self._manager.get_pool()
        expected, actual = teacher_metadata(local_client), self._pool.get_metadata()
        differences = [key for key in expected if expected[key] != actual.get(key)]
        if differences:
            raise ValueError(f"Teacher pool configuration mismatch: {', '.join(differences)}")

    def _get_tokenizer(self):
        return self.local_client._get_tokenizer()

    def get_effective_vocab_size(self):
        return self.local_client.get_effective_vocab_size()

    def get_logits(self, input_ids, attention_mask=None, prompts=None):
        ids = input_ids.detach().cpu().to(torch.int64).contiguous()
        mask = None if attention_mask is None else attention_mask.detach().cpu().bool().contiguous()
        if ids.ndim != 2 or (mask is not None and mask.shape != ids.shape):
            raise ValueError("teacher pool expects matching [batch, sequence] tensors")
        request_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.timeout
        accepted = False
        try:
            while not accepted:
                accepted = self._pool.submit(request_id, ids.numpy().tobytes(),
                                             None if mask is None else mask.numpy().tobytes(),
                                             tuple(ids.shape), prompts)
                if not accepted:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Teacher pool submission timed out under backpressure")
                    with self._lock:
                        self._stats["backpressure_waits"] += 1
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            with self._lock:
                self._stats["requests"] += 1
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Teacher pool result timed out")
                result = self._pool.get(request_id, min(1.0, remaining))
                if result is not None:
                    break
            if not result["ok"]:
                raise RuntimeError(f"Teacher pool upstream failure: {result['error']}")
            self._thread_local.metrics = result["metrics"]
            shape = tuple(result["shape"])
            if shape != (*ids.shape, self.local_client.top_k):
                raise RuntimeError("Teacher pool returned mismatched batch/sequence/top-k shape")
            return TeacherTopK(
                torch.from_numpy(np.frombuffer(result["token_ids"], dtype=np.int32).reshape(shape).copy()),
                torch.from_numpy(np.frombuffer(result["logprobs"], dtype=np.float32).reshape(shape).copy()),
            )
        except BaseException:
            if accepted:
                try:
                    self._pool.cancel(request_id)
                except Exception:
                    pass  # Preserve original failure; orphan results expire at the server.
            raise

    def get_last_request_metrics(self):
        return dict(getattr(self._thread_local, "metrics", {}))

    def get_stats(self):
        with self._lock:
            stats = dict(self._stats)
        stats.update({f"pool_{key}": value for key, value in self._pool.get_stats().items()})
        return stats

    def get_training_stats(self):
        """Keep per-step telemetry small; full counters remain available via RPC."""
        stats = self.get_stats()
        keys = ("requests", "backpressure_waits", "pool_active", "pool_queued", "pool_ready",
                "pool_failed", "pool_expired", "pool_target_workers")
        return {key: stats[key] for key in keys}
