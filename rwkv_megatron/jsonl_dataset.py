"""Small text-first JSONL datasets for online distillation smoke and training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


class JsonlTextDataset(Dataset):
    """Tokenize JSONL ``text`` records while retaining the original prompt.

    Text is split before tokenization, so a window never starts from a decoded
    byte-level token. The exact raw chunk is returned beside the padded IDs;
    the teacher therefore receives the same string that produced the student
    IDs. This is intentionally map-style for Megatron's sampler.
    """

    def __init__(self, paths: Sequence[str | Path], tokenizer, sequence_length: int,
                 target_samples: int | None = None, max_records: int = 0,
                 text_field: str = "text", chunk_chars: int = 512):
        if sequence_length < 2 or chunk_chars < 1:
            raise ValueError("sequence_length must be >= 2 and chunk_chars must be positive")
        if max_records < 0 or (target_samples is not None and target_samples < 1):
            raise ValueError("max_records must be nonnegative and target_samples positive")
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.chunk_chars = int(chunk_chars)
        self.text_field = text_field
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.samples: list[tuple[list[int], str]] = []
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
                    if not isinstance(item, dict) or not isinstance(item.get(text_field), str):
                        raise ValueError(f"JSONL record at {path}:{line_number} lacks string field {text_field!r}")
                    records += 1
                    self._add_text(item[text_field], target_samples)
                    if target_samples is not None and len(self.samples) >= target_samples:
                        break
                if target_samples is not None and len(self.samples) >= target_samples:
                    break
            if max_records and records >= max_records:
                break
        if not self.samples:
            raise ValueError(f"No usable JSONL text samples found in {[str(p) for p in paths]}")
        if target_samples is not None and len(self.samples) < target_samples:
            original = list(self.samples)
            index = 0
            while len(self.samples) < target_samples:
                self.samples.append(original[index % len(original)])
                index += 1
        if target_samples is not None:
            self.samples = self.samples[:target_samples]

    @classmethod
    def paths_for_split(cls, data_paths: Sequence[str], split: str) -> list[Path]:
        paths: list[Path] = []
        for raw in data_paths:
            path = Path(raw)
            if path.is_dir():
                candidates = sorted(path.glob("*.jsonl"))
                named = [p for p in candidates if split in p.name.lower()]
                paths.extend(named or candidates)
            elif path.suffix == ".jsonl":
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
            chunk = text[start:start + self.chunk_chars]
            self._add_chunk(chunk, target_samples)

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
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
