"""JSONL input with lazy tokenization and deterministic text window indexing."""

from __future__ import annotations

from array import array
import glob
import json
from pathlib import Path
import re
from typing import Sequence

import torch
from torch.utils.data import Dataset


class JsonlTextDataset(Dataset):
    """Keep byte/character offsets rather than corpus text or token IDs.

    Construction scans bytes to index lines, without JSON parsing or tokenizing.
    Sample offsets are filled on demand, in the original eager loader's order:
    character chunks, recursively bisected until their token IDs fit the model
    window. All usable chunks are retained. Resuming at sample N reconstructs
    the same windows regardless of access order or DataLoader worker count.

    With target_samples, len() is constant time and a short corpus repeats by
    modulo. Without a requested size, len() must finish the window index to
    determine the exact number of samples. lazy=False builds that index eagerly
    and is useful as a compatibility/reference mode.
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
        self._target_samples = target_samples
        self._paths = [Path(p) for p in paths]
        # Flat uint64 arrays: records=(file index, byte offset, byte length,
        # line number); windows=(record index, char start, char stop, token count).
        # No Python object/Path copy per line, no open handles to pickle.
        self._records = array('Q')
        self._windows = array('Q')
        self._cursor = 0
        for file_index, path in enumerate(self._paths):
            with path.open('rb') as source:
                line_number = 0
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    line_number += 1
                    if not line.strip():
                        continue
                    self._records.extend((file_index, offset, len(line), line_number))
                    if max_records and len(self._records) // 4 >= max_records:
                        break
            if max_records and len(self._records) // 4 >= max_records:
                break
        if not self._records:
            raise ValueError(f"No usable JSONL text samples found in {self._paths}")
        if not lazy:
            self._ensure_window(None if target_samples is None else target_samples - 1)

    @classmethod
    def paths_for_split(cls, data_paths: Sequence[str], split: str) -> list[Path]:
        paths = []
        aliases = {'train': {'train'}, 'validation': {'validation', 'valid', 'val'}, 'test': {'test'}}
        wanted = aliases.get(split, {split})
        known = set().union(*aliases.values())
        for raw in data_paths:
            path = Path(raw)
            if path.is_dir():
                candidates = sorted(path.rglob('*.jsonl'))
                labels = [{*p.relative_to(path).parts[:-1], *re.split(r'[-_.]', p.stem.lower())}
                          for p in candidates]
                named = [p for p, parts in zip(candidates, labels) if wanted.intersection(parts)]
                paths.extend(named if any(known.intersection(parts) for parts in labels) else candidates)
            elif glob.has_magic(str(raw)):
                paths.extend(sorted(Path(p) for p in glob.glob(str(raw), recursive=True)
                                    if Path(p).is_file() and Path(p).suffix == '.jsonl'))
            elif path.is_file() and path.suffix == '.jsonl':
                paths.append(path)
            else:
                raise ValueError(f"JSONL input must be a .jsonl file or directory: {raw}")
        if not paths:
            raise ValueError(f"No JSONL files found for split {split!r} in {data_paths}")
        return list(dict.fromkeys(p.resolve() for p in paths))

    def _read_text(self, record_index):
        file_index, offset, length, line_number = self._records[record_index * 4:record_index * 4 + 4]
        path = self._paths[file_index]
        with path.open('rb') as source:
            source.seek(offset)
            raw_line = source.read(length)
        try:
            item = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(item, dict) or not isinstance(item.get(self.text_field), str):
            raise ValueError(f"JSONL record at {path}:{line_number} lacks string field {self.text_field!r}")
        return item[self.text_field]

    def _chunk_windows(self, text, start, stop):
        encoded = self.tokenizer(text[start:stop], add_special_tokens=False).input_ids
        if len(encoded) > self.sequence_length + 1:
            if stop - start <= 1:
                raise ValueError("A single character exceeds the JSONL token window")
            middle = start + (stop - start) // 2
            yield from self._chunk_windows(text, start, middle)
            yield from self._chunk_windows(text, middle, stop)
        elif len(encoded) >= 2:
            yield start, stop, len(encoded)

    def _ensure_window(self, index):
        # Build through the requested window (or all windows for len without
        # target_samples), retaining only integer offsets, never tokenized data.
        while self._cursor < len(self._records) // 4 and (index is None or len(self._windows) // 4 <= index):
            record_index = self._cursor
            text = self._read_text(record_index)
            for start in range(0, len(text), self.chunk_chars):
                for begin, end, token_count in self._chunk_windows(text, start, min(len(text), start + self.chunk_chars)):
                    self._windows.extend((record_index, begin, end, token_count))
            self._cursor += 1
        if not self._windows:
            raise ValueError(f"No usable JSONL text samples found in {self._paths}")

    def __len__(self):
        if self._target_samples is not None:
            return self._target_samples
        self._ensure_window(None)
        return len(self._windows) // 4

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or (self._target_samples is not None and index >= self._target_samples):
            raise IndexError(index)
        return self.get_source_sample(index)

    def get_source_sample(self, index):
        """Read a raw window, extending past the logical target for replacements."""
        if index < 0:
            raise IndexError(index)
        self._ensure_window(index)
        available = len(self._windows) // 4
        if self._target_samples is None and index >= available:
            raise IndexError(index)
        index %= available
        record_index, start, stop, token_count = self._windows[index * 4:index * 4 + 4]
        prompt = self._read_text(record_index)[start:stop]
        encoded = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if len(encoded) != token_count:
            raise ValueError('JSONL contents or tokenizer changed after window indexing')
        # Fill padding by position, preserving meaningful token ID 0 and EOS
        # inside the source text even when those IDs are also used for padding.
        ids = torch.zeros(self.sequence_length + 1, dtype=torch.long)
        ids[:token_count] = torch.tensor(encoded, dtype=torch.long)
        tokens, labels = ids[:-1].clone(), ids[1:].clone()
        loss_mask = torch.zeros(self.sequence_length, dtype=torch.float32)
        loss_mask[:token_count - 1] = 1
        return dict(tokens=tokens, labels=labels, loss_mask=loss_mask,
                    position_ids=torch.arange(self.sequence_length, dtype=torch.long),
                    # Student has S inputs; when encoded has S+1 IDs, the last
                    # token is a label and the teacher client clips this text's
                    # tokenization to the same S input IDs before requesting.
                    raw_prompts=prompt)
