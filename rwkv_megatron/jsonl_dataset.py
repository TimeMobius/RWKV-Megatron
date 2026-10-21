"""Text-first JSONL datasets for online distillation.

The default loader keeps only byte offsets in memory. JSON parsing and
tokenization happen in ``__getitem__`` so large JSONL corpora do not expand to
an in-memory list of strings and token IDs during dataset construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


class JsonlTextDataset(Dataset):
    """Lazily read JSONL ``text`` records while retaining the raw prompt.

    Each JSONL record contributes one training sample. A sample is clipped to
    one sequence window when necessary; the returned ``raw_prompts`` value is
    the exact clipped text that produced ``tokens``. This keeps the online
    teacher and student autoregressive inputs aligned without decoding token
    IDs back into text.

    ``lazy=False`` retains the original eager chunking behavior for small
    compatibility experiments. Training entry points use the lazy default.
    """

    def __init__(self, paths: Sequence[str | Path], tokenizer, sequence_length: int,
                 target_samples: int | None = None, max_records: int = 0,
                 text_field: str = "text", chunk_chars: int = 512, lazy: bool = True):
        if sequence_length < 2 or chunk_chars < 1:
            raise ValueError("sequence_length must be >= 2 and chunk_chars must be positive")
        if max_records < 0 or (target_samples is not None and target_samples < 1):
            raise ValueError("max_records must be nonnegative and target_samples positive")
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.chunk_chars = int(chunk_chars)
        self.text_field = text_field
        self.lazy = bool(lazy)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self._records: list[tuple[Path, int, int]] = []
        self._target_samples = target_samples

        if self.lazy:
            self._index_lines(paths, max_records)
            self.samples = None
        else:
            self.samples: list[tuple[list[int], str]] = []
            self._load_eager(paths, target_samples, max_records)

        if not self._records and (self.lazy or not self.samples):
            raise ValueError(f"No usable JSONL text samples found in {[str(p) for p in paths]}")
        if target_samples is not None and not self.lazy:
            if len(self.samples) < target_samples:
                original = list(self.samples)
                index = 0
                while len(self.samples) < target_samples:
                    self.samples.append(original[index % len(original)])
                    index += 1
            self.samples = self.samples[:target_samples]

    def _index_lines(self, paths: Sequence[str | Path], max_records: int) -> None:
        """Build a compact ``(file, byte offset, byte length)`` line index."""
        indexed = 0
        for raw_path in paths:
            path = Path(raw_path)
            with path.open("rb") as source:
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    self._records.append((path, offset, len(line)))
                    indexed += 1
                    if max_records and indexed >= max_records:
                        return

    def _load_eager(self, paths, target_samples: int | None, max_records: int) -> None:
        records = 0
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if max_records and records >= max_records:
                        break
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
                    text = self._validate_item(item, Path(path), line_number)
                    records += 1
                    self._add_text(text, target_samples)
                    if target_samples is not None and len(self.samples) >= target_samples:
                        break
                if target_samples is not None and len(self.samples) >= target_samples:
                    break
            if max_records and records >= max_records:
                break

    def _validate_item(self, item, path: Path, line_number: int) -> str:
        if not isinstance(item, dict) or not isinstance(item.get(self.text_field), str):
            suffix = f":{line_number}" if line_number else ""
            raise ValueError(f"JSONL record at {path}{suffix} lacks string field {self.text_field!r}")
        return item[self.text_field]

    @classmethod
    def paths_for_split(cls, data_paths: Sequence[str], split: str) -> list[Path]:
        paths: list[Path] = []
        for raw in data_paths:
            path = Path(raw)
            if path.is_dir():
                candidates = sorted(path.rglob("*.jsonl"))
                named = [p for p in candidates if split.lower() in p.name.lower()]
                paths.extend(named or candidates)
            elif path.is_file() and path.suffix.lower() == ".jsonl":
                paths.append(path)
            else:
                raise ValueError(f"JSONL input must be a .jsonl file or directory: {raw}")
        if not paths:
            raise ValueError(f"No JSONL files found for split {split!r} in {data_paths}")
        return paths

    def _add_text(self, text: str, target_samples: int | None) -> None:
        if not text:
            return
        for start in range(0, len(text), self.chunk_chars):
            if target_samples is not None and len(self.samples) >= target_samples:
                return
            self._add_chunk(text[start:start + self.chunk_chars], target_samples)

    def _add_chunk(self, text: str, target_samples: int | None) -> None:
        encoded = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(encoded) > self.sequence_length + 1:
            midpoint = max(1, len(text) // 2)
            self._add_chunk(text[:midpoint], target_samples)
            self._add_chunk(text[midpoint:], target_samples)
            return
        if len(encoded) < 2:
            return
        ids = list(map(int, encoded))
        ids.extend([int(self.pad_id)] * (self.sequence_length + 1 - len(ids)))
        self.samples.append((ids, text))

    def _read_lazy_text(self, index: int) -> str:
        path, offset, length = self._records[index % len(self._records)]
        with path.open("rb") as source:
            source.seek(offset)
            raw_line = source.read(length)
        try:
            item = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSONL at {path} byte offset {offset}") from error
        return self._validate_item(item, path, 0)

    def _lazy_sample(self, index: int) -> tuple[list[int], str]:
        text = self._read_lazy_text(index)
        if not text:
            raise ValueError(f"JSONL record at sample index {index} has empty text")
        prompt = text[: self.chunk_chars]
        encoded = list(map(int, self.tokenizer(prompt, add_special_tokens=False).input_ids))
        while prompt and len(encoded) > self.sequence_length + 1:
            prompt = prompt[: max(1, len(prompt) // 2)]
            encoded = list(map(int, self.tokenizer(prompt, add_special_tokens=False).input_ids))
        if len(encoded) < 2:
            raise ValueError(f"JSONL record at sample index {index} has fewer than two tokens")
        encoded = encoded[: self.sequence_length + 1]
        encoded.extend([int(self.pad_id)] * (self.sequence_length + 1 - len(encoded)))
        return encoded, prompt

    def __len__(self):
        if self.lazy:
            return int(self._target_samples or len(self._records))
        return len(self.samples)

    def __getitem__(self, index):
        if self.lazy:
            ids, prompt = self._lazy_sample(index)
        else:
            ids, prompt = self.samples[index]
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        tokens = ids_tensor[:-1].clone()
        labels = ids_tensor[1:].clone()
        valid_labels = min(len(self.tokenizer(prompt, add_special_tokens=False).input_ids) - 1,
                           self.sequence_length)
        loss_mask = torch.zeros(self.sequence_length, dtype=torch.float32)
        loss_mask[:valid_labels] = 1.0
        labels[loss_mask == 0] = 0
        tokens[tokens == self.pad_id] = 0
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(self.sequence_length, dtype=torch.long),
            "raw_prompts": prompt,
        }
