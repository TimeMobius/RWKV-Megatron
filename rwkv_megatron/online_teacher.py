"""Online Qwen teacher support for Megatron distillation.

The teacher service is an OpenAI-compatible vLLM endpoint.  The endpoint
returns per-position ``prompt_logprobs`` when ``echo`` is enabled.  The first
position is ``null`` because there is no preceding context; positions
``1..T-1`` are aligned with the student logits at positions ``0..T-2``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


class TeacherSampleError(RuntimeError):
    """A deterministic prompt/target mismatch that another sample can replace."""


@contextmanager
def teacher_request_deadline(client, deadline):
    """Cap blocking HTTP/RPC reads on this thread at the step's deadline."""
    local = getattr(client, "_thread_local", None)
    if local is None:
        yield
        return
    previous = getattr(local, "deadline", None)
    local.deadline = deadline if previous is None else min(previous, deadline)
    try:
        yield
    finally:
        local.deadline = previous


def build_teacher_client(args, *, use_pool=True):
    """Build the same teacher configuration in training and the pool process."""
    tokenizer_path = args.rwkv_teacher_tokenizer or getattr(args, "tokenizer_model", None)
    if not tokenizer_path:
        raise ValueError("rwkv_teacher_tokenizer or tokenizer_model is required for distillation")
    client = OnlineTeacherClient(
        server_url=args.rwkv_teacher_url, model_name=args.rwkv_teacher_model,
        tokenizer_path=tokenizer_path, api_key=args.rwkv_teacher_api_key,
        top_k=args.rwkv_teacher_top_k, timeout=args.rwkv_teacher_timeout,
        request_temperature=1.0, request_seed=args.rwkv_teacher_seed,
        prompt_mode=getattr(args, "rwkv_teacher_prompt_mode", "text"),
        tokenizer_use_fast=getattr(args, "rwkv_teacher_tokenizer_use_fast", True),
        max_retries=args.rwkv_teacher_max_retries,
        retry_window=getattr(args, "rwkv_teacher_retry_window", 300.0),
        retry_backoff_initial=args.rwkv_teacher_retry_backoff_initial,
        retry_backoff_max=args.rwkv_teacher_retry_backoff_max,
        retry_jitter=args.rwkv_teacher_retry_jitter, cache_size=args.rwkv_teacher_cache_size,
    )
    if use_pool and getattr(args, "rwkv_teacher_pool", False):
        from rwkv_megatron.teacher_pool import RemoteTeacherPoolClient

        return RemoteTeacherPoolClient(
            client, host=args.rwkv_teacher_pool_host, port=args.rwkv_teacher_pool_port,
            authkey=args.rwkv_teacher_pool_authkey, timeout=args.rwkv_teacher_pool_timeout,
        )
    return client


class RoundTripDataset(torch.utils.data.Subset):
    """Keep a deterministic prefix of samples whose text preserves token IDs."""

    def __init__(self, dataset, tokenizer, count):
        selected, rejected = [], []
        for i in range(len(dataset)):
            ids = dataset[i]["tokens"].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=False)
            if tokenizer(text, add_special_tokens=False).input_ids == ids:
                selected.append(i)
                if len(selected) == count:
                    break
            else:
                rejected.append(i)
        if len(selected) != count:
            raise ValueError(f"Only {len(selected)}/{count} samples preserve teacher token IDs")
        super().__init__(dataset, selected)
        self.index_split = dataset.index_split
        self.rejected_indices = rejected


@dataclass(frozen=True)
class TeacherTopK:
    """Top-k teacher log probabilities on CPU.

    Shapes are ``[batch, sequence, k]``.  Invalid entries use token id ``-1``
    and log probability ``-inf``.
    """

    token_ids: torch.Tensor
    logprobs: torch.Tensor


class TeacherPrefetchIterator:
    """Bounded, ordered CPU batches with concurrent teacher requests.

    Only the training thread advances the source iterator. Workers receive
    private CPU token copies and never touch CUDA or Megatron state. The
    returned future stays paired with its batch even if HTTP calls finish out
    of order. Resume must reconstruct the sampler at *consumed* samples, as
    Megatron's single dataloader does; pending batches are disposable.
    """

    future_key = "_rwkv_teacher_future"

    def __init__(self, iterable, teacher_client, num_workers=16, lookahead=32, max_batches=None,
                 min_workers=1, latency_target_seconds=2.0):
        if num_workers < 1 or lookahead < 1:
            raise ValueError("teacher workers and lookahead must be positive")
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be nonnegative")
        if not 1 <= min_workers <= num_workers:
            raise ValueError("min_workers must be in [1, num_workers]")
        if latency_target_seconds <= 0 or not math.isfinite(latency_target_seconds):
            raise ValueError("latency_target_seconds must be positive")
        self.iterable = iter(iterable)
        self.teacher_client = teacher_client
        self.lookahead = lookahead
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="qwen-teacher")
        self._pending = deque()
        self._exhausted = False
        self._remaining = max_batches
        self._closed = False
        self._lock = threading.Lock()
        self._concurrency = threading.Condition(self._lock)
        self._max_workers = num_workers
        self._min_workers = min_workers
        self._target_workers = num_workers
        self._latency_target_seconds = float(latency_target_seconds)
        self._stats = dict(submitted=0, consumed=0, completed=0, failed=0, active=0,
                           peak_active=0, target_workers=num_workers, concurrency_reductions=0,
                           concurrency_increases=0, observed_retries=0, observed_rate_limits=0)

    def __iter__(self):
        return self

    def _fetch(self, tokens, prompts=None, attention_mask=None):
        with self._concurrency:
            while self._stats["active"] >= self._target_workers and not self._closed:
                self._concurrency.wait(timeout=1.0)
            self._stats["active"] += 1
            self._stats["peak_active"] = max(self._stats["peak_active"], self._stats["active"])
        try:
            # Token-ID mode must receive the right-padded mask so padded rows
            # are not sent as part of the vLLM prompt.  The dataset's loss mask
            # is converted before submission in ``__next__``.
            if getattr(self.teacher_client, "prompt_mode", "text") == "token_ids":
                result = self.teacher_client.get_logits(tokens, attention_mask=attention_mask)
            else:
                result = (self.teacher_client.get_logits(tokens, prompts=prompts)
                          if prompts is not None else self.teacher_client.get_logits(tokens))
            metrics = getattr(self.teacher_client, "get_last_request_metrics", lambda: {})()
            self._observe_request(metrics)
        except BaseException:
            with self._lock:
                self._stats["failed"] += 1
            raise
        else:
            with self._lock:
                self._stats["completed"] += 1
            return result
        finally:
            with self._concurrency:
                self._stats["active"] -= 1
                self._concurrency.notify_all()

    def _observe_request(self, metrics):
        """React to service pressure without creating more executor threads."""
        if not metrics:
            return
        with self._concurrency:
            retries = int(metrics.get("retries", 0))
            rate_limits = int(metrics.get("rate_limits", 0))
            latency = float(metrics.get("latency_seconds", 0.0))
            self._stats["observed_retries"] += retries
            self._stats["observed_rate_limits"] += rate_limits
            old = self._target_workers
            if rate_limits:
                self._target_workers = max(self._min_workers, max(1, old // 2))
            elif latency > self._latency_target_seconds and old > self._min_workers:
                self._target_workers = old - 1
            elif latency and latency < self._latency_target_seconds * 0.5 and old < self._max_workers:
                self._target_workers = old + 1
            if self._target_workers != old:
                key = "concurrency_increases" if self._target_workers > old else "concurrency_reductions"
                self._stats[key] += 1
                self._stats["target_workers"] = self._target_workers
                self._concurrency.notify_all()

    def __next__(self):
        if self._closed:
            raise StopIteration
        # Fill before returning: API work for this and later batches overlaps
        # the student's forward/backward, without a background dataloader.
        while len(self._pending) < self.lookahead and not self._exhausted:
            if self._remaining == 0:
                self._exhausted = True
                break
            try:
                batch = next(self.iterable)
            except StopIteration:
                self._exhausted = True
                break
            if batch["tokens"].device.type != "cpu":
                raise ValueError("Teacher prefetch requires CPU dataloader tokens")
            tokens = batch["tokens"].detach().clone()
            prompts = batch.get("raw_prompts")
            if "teacher_token_ids" in batch:
                future = Future()
                future.set_result(TeacherTopK(batch["teacher_token_ids"], batch["teacher_logprobs"]))
                with self._lock:
                    self._stats["completed"] += 1
            elif getattr(self.teacher_client, "prompt_mode", "text") == "token_ids":
                attention_mask = None
                if prompts is not None:
                    loss_mask = batch.get("loss_mask")
                    if loss_mask is None:
                        raise ValueError("JSONL token-ID teacher prefetch requires batch loss_mask")
                    lengths = (loss_mask.detach().sum(dim=-1).to(torch.long) + 1).clamp_max(tokens.shape[-1])
                    attention_mask = torch.arange(tokens.shape[-1]).unsqueeze(0) < lengths.unsqueeze(1)
                future = self._executor.submit(self._fetch, tokens, prompts, attention_mask)
            else:
                future = self._executor.submit(self._fetch, tokens, prompts)
            self._pending.append((batch, future))
            if self._remaining is not None:
                self._remaining -= 1
            with self._lock:
                self._stats["submitted"] += 1
        if not self._pending:
            raise StopIteration
        batch, future = self._pending.popleft()
        with self._lock:
            self._stats["consumed"] += 1
        return {**batch, self.future_key: future}

    def get_stats(self):
        with self._lock:
            result = dict(self._stats, buffered=len(self._pending))
        client_stats = getattr(self.teacher_client, "get_training_stats",
                               getattr(self.teacher_client, "get_stats", lambda: {}))()
        result.update({f"client_{key}": value for key, value in client_stats.items()})
        return result

    def close(self):
        """Cancel queued work and drain in-flight HTTP calls (client timeout bounds them)."""
        if not self._closed:
            self._closed = True
            for _, future in self._pending:
                future.cancel()
            self._pending.clear()
            with self._concurrency:
                self._concurrency.notify_all()
            self._executor.shutdown(wait=True, cancel_futures=True)
            # Release the DataLoader iterator in the main thread. Keeping it
            # alive in a global prefetcher prevents its worker shutdown before
            # multiprocessing's interpreter-exit joins and can hang exit.
            self.iterable = None


class OnlineTeacherClient:
    """Fetch Qwen prompt log probabilities from a vLLM HTTP service."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        tokenizer_path: str,
        top_k: int = 100,
        timeout: float = 600.0,
        request_temperature: float = 1.0,
        request_seed: Optional[int] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_initial: float = 1.0,
        retry_backoff_max: float = 30.0,
        retry_jitter: float = 0.2,
        cache_size: int = 256,
        prompt_mode: str = "text",
        tokenizer_use_fast: bool = True,
        retry_window: float = 0.0,
    ) -> None:
        if not 1 <= int(top_k) <= 1000:
            raise ValueError("top_k must be in [1, 1000] for the teacher service")
        if not math.isfinite(request_temperature) or request_temperature <= 0:
            raise ValueError("request_temperature must be positive")
        self.server_url = server_url.rstrip("/").removesuffix("/v1")
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path
        self.top_k = int(top_k)
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("teacher timeout must be finite and positive")
        if not math.isfinite(retry_window) or retry_window < 0:
            raise ValueError("teacher retry window must be finite and nonnegative")
        self.retry_window = float(retry_window)
        self.request_temperature = float(request_temperature)
        if prompt_mode not in ("text", "token_ids"):
            raise ValueError("prompt_mode must be 'text' or 'token_ids'")
        self.prompt_mode = prompt_mode
        if type(tokenizer_use_fast) is not bool:
            raise ValueError("tokenizer_use_fast must be a boolean")
        self.tokenizer_use_fast = tokenizer_use_fast
        if request_seed is not None and (type(request_seed) is not int or request_seed < 0):
            raise ValueError("request_seed must be a nonnegative integer or None")
        self.request_seed = request_seed
        if max_retries < 0 or retry_backoff_initial < 0 or retry_backoff_max < retry_backoff_initial or retry_jitter < 0:
            raise ValueError("invalid teacher retry settings")
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.max_retries = int(max_retries)
        self.retry_backoff_initial = float(retry_backoff_initial)
        self.retry_backoff_max = float(retry_backoff_max)
        self.retry_jitter = float(retry_jitter)
        self.cache_size = int(cache_size)
        self._api_key = api_key if api_key is not None else os.environ.get("RWKV_TEACHER_API_KEY", "")
        self._thread_local = threading.local()
        self._tokenizer = None
        self._effective_vocab_size = None
        self._tokenizer_lock = threading.Lock()
        self._calls_lock = threading.Lock()
        self._calls = 0
        self._stats_lock = threading.Lock()
        self._stats = dict(requests=0, retries=0, rate_limits=0, failures=0, cache_hits=0,
                           cache_entries=0, ewma_latency_seconds=0.0)
        self._cache_lock = threading.Lock()
        self._cache = OrderedDict()

    def _get_session(self):
        import requests

        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            if self._api_key:
                session.headers["Authorization"] = f"Bearer {self._api_key}"
            self._thread_local.session = session
        return session

    def _get_tokenizer(self):
        with self._tokenizer_lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    trust_remote_code=True,
                    use_fast=self.tokenizer_use_fast,
                )
                # ``tokenizer.vocab_size`` is the size of the base vocabulary
                # for Qwen and excludes added/special tokens.  The teacher
                # service can return any ID in the full tokenizer vocabulary,
                # so derive the usable upper bound from the actual IDs.
                vocabulary = self._tokenizer.get_vocab()
                if not vocabulary:
                    raise ValueError(f"Tokenizer at {self.tokenizer_path!r} has an empty vocabulary")
                self._effective_vocab_size = max(int(token_id) for token_id in vocabulary.values()) + 1
        return self._tokenizer

    def get_effective_vocab_size(self) -> int:
        """Return the highest valid tokenizer ID plus one.

        Megatron pads the student output vocabulary to a divisible size.  This
        value lets the distillation loss exclude those extra rows from the
        student softmax denominator.
        """

        self._get_tokenizer()
        return int(self._effective_vocab_size)

    def _decode(self, token_ids: list[int]) -> str:
        return self._get_tokenizer().decode(token_ids, skip_special_tokens=False)

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompts: Optional[list[str] | tuple[str, ...] | str] = None,
    ) -> TeacherTopK:
        """Return teacher top-k prompt distributions for a batch.

        Requests are intentionally made with temperature 1.0 by default.  The
        distillation temperature is applied only by the local KL loss.
        """

        input_ids = input_ids.detach().to(device="cpu", dtype=torch.long)
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, sequence], got {tuple(input_ids.shape)}")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.detach().to(device="cpu", dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"input_ids {tuple(input_ids.shape)}"
            )
        prompt_texts = None
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            prompt_texts = list(prompts)
            if len(prompt_texts) != input_ids.shape[0]:
                raise ValueError("prompts must contain one string per input row")

        cache_key = self._cache_key(input_ids, attention_mask)
        if self.cache_size:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    with self._stats_lock:
                        self._stats["cache_hits"] += 1
                    self._thread_local.last_request_metrics = dict(
                        latency_seconds=0.0, retries=0, rate_limits=0, cache_hit=True
                    )
                    return TeacherTopK(cached.token_ids.clone(), cached.logprobs.clone())

        request_start = time.monotonic()
        total_retries = 0
        total_rate_limits = 0
        prompt_rows = []
        for row_index, (row, mask) in enumerate(zip(input_ids, attention_mask)):
            prompt = prompt_texts[row_index] if prompt_texts is not None else None
            valid_len = int(mask.sum().item())
            if not mask[:valid_len].all() or mask[valid_len:].any():
                raise ValueError("Online teacher requires right-padded inputs")
            row_ids = row[:valid_len].tolist()
            if self.prompt_mode == "token_ids":
                # vLLM accepts a list of token IDs as the completion prompt.
                # This path never decodes or re-tokenizes the input.
                prompt_rows.append((row_ids, valid_len, row_ids))
                continue
            if prompt is None:
                prompt = self._decode(row_ids) if valid_len else ""
            else:
                row_ids_from_text = self._get_tokenizer()(prompt, add_special_tokens=False).input_ids
                valid_len = len(row_ids_from_text)
                # A dataset window may contain S+1 IDs: S inputs plus its last
                # next-token label. The teacher can score that final label too;
                # only its first S prompt positions are returned to the loss.
                if valid_len > row.shape[0] + 1 or row[:valid_len].tolist() != row_ids_from_text[:row.shape[0]]:
                    raise TeacherSampleError("Raw JSONL prompt does not match the student token IDs")
                row_ids = row_ids_from_text
            if valid_len and self._get_tokenizer()(prompt, add_special_tokens=False).input_ids != row_ids:
                raise TeacherSampleError("Teacher prompt does not round-trip to the student token IDs")
            prompt_rows.append((prompt, valid_len, row_ids))

        token_rows = []
        logprob_rows = []
        for prompt, valid_len, row_ids in prompt_rows:
            token_ids, logprobs, request_metrics = self._request_prompt_logprobs(
                prompt, valid_len, return_metrics=True, expected_token_ids=row_ids
            )
            total_retries += request_metrics["retries"]
            total_rate_limits += request_metrics["rate_limits"]
            token_ids = token_ids[: input_ids.shape[1]]
            logprobs = logprobs[: input_ids.shape[1]]
            valid_len = min(valid_len, input_ids.shape[1])
            padding = input_ids.shape[1] - valid_len
            if padding:
                token_ids = torch.cat(
                    (
                        token_ids,
                        torch.full((padding, self.top_k), -1, dtype=torch.int32),
                    )
                )
                logprobs = torch.cat(
                    (logprobs, torch.full((padding, self.top_k), -torch.inf))
                )
            token_rows.append(token_ids)
            logprob_rows.append(logprobs)
        result = TeacherTopK(torch.stack(token_rows), torch.stack(logprob_rows))
        latency = time.monotonic() - request_start
        self._record_request_metrics(latency, total_retries, total_rate_limits)
        self._thread_local.last_request_metrics = dict(
            latency_seconds=latency, retries=total_retries, rate_limits=total_rate_limits, cache_hit=False
        )
        if self.cache_size:
            with self._cache_lock:
                self._cache[cache_key] = TeacherTopK(result.token_ids.clone(), result.logprobs.clone())
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
                with self._stats_lock:
                    self._stats["cache_entries"] = len(self._cache)
        return result

    def _request_prompt_logprobs(self, prompt: str | list[int], sequence_length: int,
                                 return_metrics: bool = False, expected_token_ids=None):
        token_ids = torch.full((sequence_length, self.top_k), -1, dtype=torch.int32)
        logprobs = torch.full((sequence_length, self.top_k), float("-inf"), dtype=torch.float32)
        if not prompt:
            result = (token_ids, logprobs, {"retries": 0, "rate_limits": 0})
            return result if return_metrics else result[:2]

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": self.request_temperature,
            "top_p": 1.0,
            "logprobs": self.top_k,
            "prompt_logprobs": self.top_k,
            "echo": True,
        }
        if self.request_seed is not None:
            payload["seed"] = self.request_seed
        import requests

        retries = 0
        rate_limits = 0
        deadline = getattr(self._thread_local, "deadline", None)
        if self.retry_window:
            window_deadline = time.monotonic() + self.retry_window
            deadline = window_deadline if deadline is None else min(deadline, window_deadline)
        attempt = 0
        last_error = None
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                with self._stats_lock:
                    self._stats["failures"] += 1
                detail = ""
                if last_error is not None:
                    detail = f"; last error: {type(last_error).__name__}"
                    if isinstance(last_error, _RetryableTeacherError):
                        detail += f" HTTP {last_error.status}"
                raise TimeoutError("Teacher request retry window/deadline exhausted" + detail) from last_error
            with self._calls_lock:
                self._calls += 1
            try:
                response = self._get_session().post(
                    f"{self.server_url}/v1/completions",
                    json=payload,
                    timeout=self.timeout if remaining is None else min(self.timeout, remaining),
                )
                status = int(getattr(response, "status_code", 200))
                if status == 429:
                    rate_limits += 1
                    raise _RetryableTeacherError(status, getattr(response, "headers", {}).get("Retry-After"))
                if status in (408, 425) or status >= 500:
                    raise _RetryableTeacherError(status, getattr(response, "headers", {}).get("Retry-After"))
                response.raise_for_status()
                result = response.json()
                parsed = self._parse_response(
                    result, sequence_length, token_ids, logprobs, expected_token_ids
                )
                result = (
                    parsed[0],
                    parsed[1],
                    {"retries": retries, "rate_limits": rate_limits},
                )
                return result if return_metrics else result[:2]
            except TeacherSampleError:
                raise
            except (
                requests.RequestException,
                _RetryableTeacherError,
                RuntimeError,
            ) as error:
                if isinstance(error, requests.HTTPError):
                    status = getattr(
                        getattr(error, "response", None), "status_code", None
                    )
                    if status not in (408, 425, 429) and not (
                        status is not None and status >= 500
                    ):
                        with self._stats_lock:
                            self._stats["failures"] += 1
                        raise
                transient = isinstance(error, (requests.ConnectionError, requests.Timeout,
                                               _RetryableTeacherError))
                if attempt >= self.max_retries and not (self.retry_window and transient):
                    with self._stats_lock:
                        self._stats["failures"] += 1
                    raise
                retry_after = getattr(error, "retry_after", None)
                try:
                    retry_after = float(retry_after) if retry_after is not None else 0.0
                except (TypeError, ValueError):
                    retry_after = 0.0
                delay = (
                    retry_after
                    if retry_after > 0
                    else min(
                        self.retry_backoff_max,
                        self.retry_backoff_initial * (2**attempt),
                    )
                )
                delay += random.uniform(0.0, self.retry_jitter)
                if deadline is not None:
                    delay = min(delay, max(0.0, deadline - time.monotonic()))
                retries += 1
                with self._stats_lock:
                    self._stats["retries"] += 1
                    self._stats["rate_limits"] += int(
                        isinstance(error, _RetryableTeacherError)
                        and error.status == 429
                    )
                logging.getLogger(__name__).warning(
                    "Teacher temporarily unavailable (%s); retry %s in %.1fs%s",
                    (f"HTTP {error.status}" if isinstance(error, _RetryableTeacherError)
                     else type(error).__name__), retries, delay,
                    " (bounded recovery window)" if self.retry_window else "",
                )
                last_error = error
                time.sleep(delay)
                attempt += 1

    def _parse_response(
        self, result, sequence_length, token_ids, logprobs, expected_token_ids=None
    ):
        choices = result.get("choices") or []
        choice = choices[0] if choices else {}
        prompt_logprobs = choice.get("prompt_logprobs")
        if not isinstance(prompt_logprobs, list):
            raise TeacherSampleError(
                "Teacher response does not contain prompt_logprobs; "
                "set echo=true and prompt_logprobs in the vLLM service"
            )

        if len(prompt_logprobs) != sequence_length:
            raise TeacherSampleError(
                f"Teacher returned {len(prompt_logprobs)} prompt positions for {sequence_length} tokens; "
                f"check teacher tokenizer implementation (tokenizer_use_fast={self.tokenizer_use_fast})"
            )

        if expected_token_ids is not None:
            actual_ids = choice.get("prompt_token_ids")
            if isinstance(actual_ids, list):
                if actual_ids != expected_token_ids:
                    raise TeacherSampleError(
                        "Teacher prompt token IDs differ from student token IDs"
                    )

        for position, position_values in enumerate(prompt_logprobs[:sequence_length]):
            if not isinstance(position_values, dict):
                continue
            row_ids, row_logprobs = [], []
            for column, (token_id, value) in enumerate(position_values.items()):
                if column >= self.top_k:
                    break
                try:
                    token_id = int(token_id)
                    value = value.get("logprob") if isinstance(value, dict) else value
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                row_ids.append(token_id)
                row_logprobs.append(value)
            # Construct each row together; per-element tensor assignments
            # serialize CPU parsing across concurrent Python HTTP workers.
            if row_ids:
                token_ids[position, :len(row_ids)] = torch.tensor(row_ids, dtype=torch.int32)
                logprobs[position, :len(row_logprobs)] = torch.tensor(row_logprobs, dtype=torch.float32)
        if sequence_length > 1 and not torch.isfinite(logprobs[1:]).any(dim=-1).all():
            raise TeacherSampleError("Teacher returned missing or nonfinite prompt log probabilities")
        return token_ids, logprobs

    @staticmethod
    def _cache_key(input_ids, attention_mask):
        digest = hashlib.blake2b(digest_size=20)
        digest.update(input_ids.contiguous().numpy().tobytes())
        digest.update(attention_mask.contiguous().numpy().tobytes())
        return digest.digest()

    def _record_request_metrics(self, latency, retries, rate_limits):
        with self._stats_lock:
            self._stats["requests"] += 1
            old = self._stats["ewma_latency_seconds"]
            self._stats["ewma_latency_seconds"] = latency if old == 0 else 0.2 * latency + 0.8 * old

    def get_last_request_metrics(self):
        return dict(getattr(self._thread_local, "last_request_metrics", {}))

    def get_stats(self):
        with self._stats_lock:
            return dict(self._stats)


class _RetryableTeacherError(RuntimeError):
    def __init__(self, status, retry_after=None):
        super().__init__(f"retryable teacher HTTP status {status}")
        self.status = status
        self.retry_after = retry_after


def kl_divergence_topk(
    student_logits: torch.Tensor,
    teacher: TeacherTopK,
    attention_mask: torch.Tensor,
    temperature: float = 2.0,
    valid_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute top-k KL for already aligned student/teacher positions.

    ``student_logits`` has shape ``[B, T, V]`` and teacher tensors have shape
    ``[B, T, K]``.  The teacher top-k mass is renormalized before computing KL,
    matching the source distiller's behavior.
    """

    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape [batch, sequence, vocab]")
    if teacher.token_ids.ndim != 3 or teacher.logprobs.ndim != 3:
        raise ValueError("teacher top-k tensors must have shape [batch, sequence, k]")
    if teacher.token_ids.shape != teacher.logprobs.shape:
        raise ValueError("teacher token_ids and logprobs must have the same shape")
    if teacher.token_ids.shape[:2] != student_logits.shape[:2]:
        raise ValueError("teacher and student must have the same batch/sequence shape")
    if attention_mask.shape != student_logits.shape[:2]:
        raise ValueError("attention_mask must match the aligned student sequence")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")

    # Reject incompatible vocabularies before CUDA gather. Even equal numeric
    # IDs are meaningful only when both models use the same tokenizer.
    student_vocab_size = student_logits.shape[-1]
    if valid_vocab_size is None:
        valid_vocab_size = student_vocab_size
    if not isinstance(valid_vocab_size, int) or not 1 <= valid_vocab_size <= student_vocab_size:
        raise ValueError(
            f"valid_vocab_size must be an integer in [1, {student_vocab_size}], "
            f"got {valid_vocab_size!r}"
        )
    if (teacher.token_ids >= valid_vocab_size).any() or (teacher.token_ids < -1).any():
        raise ValueError("Teacher token IDs exceed student vocabulary; use the same tokenizer")
    # The final rows are Megatron padding (248077..248319 for Qwen).  They
    # have no teacher targets and must not participate in the student
    # softmax denominator, otherwise they absorb probability mass and change
    # both the KL value and gradients for every valid token.
    student = student_logits[..., :valid_vocab_size].float() / float(temperature)
    token_ids = teacher.token_ids.to(device=student.device, dtype=torch.long)
    teacher_logprobs = teacher.logprobs.to(device=student.device, dtype=torch.float32) / float(temperature)
    valid = attention_mask.to(device=student.device, dtype=torch.bool)
    token_valid = (token_ids >= 0) & torch.isfinite(teacher_logprobs)
    teacher_logprobs = teacher_logprobs.masked_fill(~token_valid, -torch.inf)
    teacher_log_z = torch.logsumexp(teacher_logprobs, dim=-1, keepdim=True)
    valid = valid & torch.isfinite(teacher_log_z.squeeze(-1))
    if not valid.any():
        return student_logits.float().sum() * 0.0

    # Select usable rows before normalization so absent/masked teacher rows
    # never introduce NaNs into the loss or its gradient.
    student = student[valid]
    token_ids = token_ids[valid]
    token_valid = token_valid[valid]
    teacher_logprobs = teacher_logprobs[valid]
    teacher_log_z = teacher_log_z[valid]
    teacher_logp = teacher_logprobs - teacher_log_z
    teacher_prob = torch.exp(teacher_logp)
    safe_ids = token_ids.clamp_min(0)
    student_logp_at = student.gather(-1, safe_ids) - torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_prob = teacher_prob.masked_fill(~token_valid, 0.0)
    teacher_logp = teacher_logp.masked_fill(~token_valid, 0.0)
    student_logp_at = student_logp_at.masked_fill(~token_valid, 0.0)
    per_token = (teacher_prob * (teacher_logp - student_logp_at)).sum(dim=-1)
    return per_token.mean() * float(temperature) ** 2
