"""Select the next valid JSONL window before any pipeline stage runs forward."""

import hashlib
import json
import logging
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .online_teacher import TeacherSampleError, TeacherTopK


def padding_targets(sample, teacher):
    client = getattr(teacher, 'local_client', teacher)
    shape = (1, sample['tokens'].numel(), getattr(client, 'top_k', 1))
    return TeacherTopK(torch.full(shape, -1, dtype=torch.int32),
                       torch.full(shape, -torch.inf, dtype=torch.float32))


class TeacherValidatedDataset(Dataset):
    """Logical index N is the Nth valid source window, across DP/PP and resume.

    The shared decision journal stores selection metadata, not teacher tensors.
    Every reader reconstructs the same valid prefix without re-requesting old
    decisions. Lookahead validates a bounded number of future source windows.
    """

    def __init__(self, source, teacher, directory, signature, lookahead=4):
        self.source, self.teacher = source, teacher
        if type(lookahead) is not int or lookahead < 1:
            raise ValueError("lookahead must be a positive integer")
        self.lookahead = lookahead
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(signature.encode()).hexdigest()
        self.db = self.directory / (digest + ".sqlite")
        self.log = self.directory / "rejected-samples.jsonl"
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS decisions (source_index INTEGER PRIMARY KEY, sample_hash TEXT, valid INTEGER, record TEXT)"
            )
        self.selected = []
        self.next_source = 0
        self.pending = deque()
        self.executor = None
        self.ready = {}

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db, timeout=60)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def __len__(self):
        return len(self.source)

    def _decision(self, index, sample):
        if sample.get('epoch_padding', False):
            return True, padding_targets(sample, self.teacher)
        sample_hash = hashlib.sha256(
            sample["tokens"].numpy().tobytes() + sample["raw_prompts"].encode()
        ).hexdigest()
        with self._connection() as connection:
            previous = connection.execute(
                "SELECT sample_hash, valid FROM decisions WHERE source_index=?",
                (index,),
            ).fetchone()
        if previous:
            if previous[0] != sample_hash:
                raise ValueError(
                    "Source data/tokenizer changed after teacher sample selection"
                )
            return bool(previous[1]), None
        reason = None
        try:
            kwargs = {"prompts": [sample["raw_prompts"]]}
            if getattr(self.teacher, "prompt_mode", "text") == "token_ids":
                count = min(
                    int(sample["loss_mask"].sum()) + 1, sample["tokens"].numel()
                )
                kwargs = {
                    "attention_mask": (
                        torch.arange(sample["tokens"].numel()) < count
                    ).unsqueeze(0)
                }
            targets = self.teacher.get_logits(sample["tokens"].unsqueeze(0), **kwargs)
        except TeacherSampleError as error:
            reason = str(error)
        valid = reason is None
        record = json.dumps(
            dict(
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_index=index,
                sample_sha256=sample_hash,
                reason=reason,
                selection_namespace=self.db.stem,
                raw_prompt=sample["raw_prompts"],
                token_ids=sample["tokens"].tolist(),
            ),
            ensure_ascii=False,
        )
        with self._connection() as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO decisions VALUES (?, ?, ?, ?)",
                (index, sample_hash, int(valid), record),
            ).rowcount
            previous = connection.execute(
                "SELECT sample_hash, valid FROM decisions WHERE source_index=?",
                (index,),
            ).fetchone()
            if previous != (sample_hash, int(valid)):
                raise RuntimeError(
                    "Teacher sample validity changed between pipeline/DP readers"
                )
        if inserted and not valid:
            with self.log.open("a") as stream:
                stream.write(record + "\n")
            logging.getLogger(__name__).warning(
                "Rejecting source sample %s (%s): %s; trying next sample",
                index,
                sample_hash,
                reason,
            )
        return valid, targets if valid else None

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=self.lookahead, thread_name_prefix="teacher-validation"
            )
        rejected = 0
        while len(self.selected) <= index:
            while len(self.pending) < min(self.lookahead, len(self) - len(self.selected)):
                raw = self.next_source
                sample = self.source.get_source_sample(raw)
                self.pending.append(
                    (raw, self.executor.submit(self._decision, raw, sample))
                )
                self.next_source += 1
            raw, future = self.pending.popleft()
            valid, targets = future.result()
            if valid:
                if targets is not None and len(self.selected) == index:
                    self.ready[raw] = targets
                self.selected.append(raw)
                rejected = 0
            else:
                rejected += 1
                if rejected >= 100:
                    raise RuntimeError(
                        "100 consecutive teacher samples failed validation; check the teacher/tokenizer configuration"
                    )
        raw = self.selected[index]
        sample = self.source.get_source_sample(raw)
        targets = self.ready.pop(raw, None)
        if sample.get('epoch_padding', False):
            targets = padding_targets(sample, self.teacher)
        if targets is None:
            kwargs = {"prompts": [sample["raw_prompts"]]}
            if getattr(self.teacher, "prompt_mode", "text") == "token_ids":
                count = min(
                    int(sample["loss_mask"].sum()) + 1, sample["tokens"].numel()
                )
                kwargs = {
                    "attention_mask": (
                        torch.arange(sample["tokens"].numel()) < count
                    ).unsqueeze(0)
                }
            targets = self.teacher.get_logits(sample["tokens"].unsqueeze(0), **kwargs)
        # The exact validated response travels with the selected input, so the
        # loss never issues a second request after the student has run forward.
        return dict(
            sample,
            teacher_token_ids=targets.token_ids[0].cpu(),
            teacher_logprobs=targets.logprobs[0].cpu(),
        )

    def rejected_count(self):
        with self._connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM decisions WHERE valid=0"
            ).fetchone()[0]

    def close(self):
        if self.executor is not None:
            for _, future in self.pending:
                future.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.pending.clear()
            self.next_source = self.selected[-1] + 1 if self.selected else 0


def selection_signature(source, args, split):
    files = [
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in source._paths
    ]
    tokenizer = source.tokenizer
    vocabulary_hash = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    ).hexdigest()
    return json.dumps(
        dict(
            files=files,
            split=split,
            sequence=source.sequence_length,
            chunk_chars=source.chunk_chars,
            text_field=source.text_field,
            max_records=args.rwkv_jsonl_max_records,
            tokenizer=type(tokenizer).__name__,
            vocabulary=vocabulary_hash,
            teacher=args.rwkv_teacher_model,
            url=args.rwkv_teacher_url,
            top_k=args.rwkv_teacher_top_k,
            seed=args.rwkv_teacher_seed,
            prompt_mode=args.rwkv_teacher_prompt_mode,
        ),
        sort_keys=True,
    )
