import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from rwkv_megatron.jsonl_dataset import JsonlTextDataset
from rwkv_megatron.online_teacher import OnlineTeacherClient


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) % 31 + 1 for char in text])


def test_jsonl_dataset_keeps_raw_text_and_masks_padding(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps({"text": "abcdef"}) + "\n", encoding="utf-8")
    dataset = JsonlTextDataset([path], TinyTokenizer(), sequence_length=8, target_samples=2,
                               chunk_chars=32)
    first = dataset[0]
    assert first["raw_prompts"] == "abcdef"
    assert first["tokens"].tolist() == [5, 6, 7, 8, 9, 10, 0, 0]
    assert first["labels"].tolist() == [6, 7, 8, 9, 10, 0, 0, 0]
    assert first["loss_mask"].tolist() == [1.0] * 5 + [0.0] * 3
    assert len(dataset) == 2


def test_jsonl_paths_select_split_files(tmp_path):
    nested = tmp_path / "shards" / "nested"
    nested.mkdir(parents=True)
    for name in ("train-0.jsonl", "validation-0.jsonl", "test-0.jsonl"):
        (nested / name).write_text('{"text":"ok"}\n')
    assert JsonlTextDataset.paths_for_split([tmp_path], "validation")[0].name == "validation-0.jsonl"


def test_jsonl_loader_indexes_without_tokenizing_until_getitem(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"text":"abcdef"}\n{"text":"ghijkl"}\n', encoding="utf-8")

    class CountingTokenizer(TinyTokenizer):
        def __init__(self):
            self.calls = 0

        def __call__(self, text, add_special_tokens=False):
            self.calls += 1
            return super().__call__(text, add_special_tokens=add_special_tokens)

    tokenizer = CountingTokenizer()
    dataset = JsonlTextDataset([path], tokenizer, sequence_length=8, target_samples=4)
    assert tokenizer.calls == 0
    assert len(dataset) == 4
    assert dataset[2]["raw_prompts"] == "abcdef"
    assert tokenizer.calls > 0


def test_lazy_windows_match_original_eager_algorithm_and_keep_document_tails(tmp_path):
    texts = ['a', '', 'abcdefghij' * 4, 'uvwxyz', '中文测试字符' * 5]
    path = tmp_path / 'train.jsonl'
    path.write_text(''.join(json.dumps({'text': t}) + '\n' for t in texts))
    tokenizer = TinyTokenizer()
    # Independent reference to the original text splitting policy.
    expected = []
    def split_chunk(text):
        ids = tokenizer(text).input_ids
        if len(ids) > 9:
            mid = max(1, len(text) // 2)
            split_chunk(text[:mid])
            split_chunk(text[mid:])
        elif len(ids) >= 2:
            expected.append(text)
    for text in texts:
        for start in range(0, len(text), 13):
            split_chunk(text[start:start + 13])
    dataset = JsonlTextDataset([path], tokenizer, 8, target_samples=len(expected) * 2, chunk_chars=13)
    # Random access forces index construction; subsequent sequential access is
    # identical, including repeated epochs and the final chunk of each record.
    assert dataset[len(expected) - 1]['raw_prompts'] == expected[-1]
    for i in reversed(range(len(dataset))):
        sample = dataset[i]
        prompt = expected[i % len(expected)]
        assert sample['raw_prompts'] == prompt
        ids = tokenizer(prompt).input_ids
        assert sample['tokens'][:min(len(ids), 8)].tolist() == ids[:8]
        assert sample['labels'][:len(ids)-1].tolist() == ids[1:]
        assert sample['loss_mask'].sum() == len(ids) - 1
    full = JsonlTextDataset([path], tokenizer, 8, chunk_chars=13)
    assert len(full) == len(expected)
    with pytest.raises(IndexError):
        full[len(full)]


def test_lazy_multiworker_resume_is_identical(tmp_path):
    from torch.utils.data import DataLoader, Subset
    path = tmp_path / 'train.jsonl'
    path.write_text(''.join(json.dumps({'text': 'abcdefghijklmnop' * i}) + '\n' for i in range(1, 6)))
    ds = JsonlTextDataset([path], TinyTokenizer(), 8, target_samples=30, chunk_chars=13)
    expected = [ds[i] for i in range(30)]
    fresh = JsonlTextDataset([path], TinyTokenizer(), 8, target_samples=30, chunk_chars=13)
    rows = list(DataLoader(Subset(fresh, range(13, 30)), batch_size=None, num_workers=2))
    for actual, reference in zip(rows, expected[13:]):
        assert actual['raw_prompts'] == reference['raw_prompts']
        for key in ('tokens', 'labels', 'loss_mask'):
            assert torch.equal(actual[key], reference[key])


def test_recursive_split_subdirectories_globs_and_dedup(tmp_path):
    for split in ('train', 'validation', 'test'):
        folder = tmp_path / split / 'nested'
        folder.mkdir(parents=True)
        (folder / 'part-0.jsonl').write_text('{"text":"abc"}\n')
    selected = JsonlTextDataset.paths_for_split([tmp_path], 'validation')
    assert len(selected) == 1 and 'validation' in selected[0].parts
    all_files = JsonlTextDataset.paths_for_split([str(tmp_path / '**/*.jsonl')], 'train')
    assert len(all_files) == 3
    assert JsonlTextDataset.paths_for_split([tmp_path, selected[0]], 'validation') == selected


def test_invalid_json_is_reported_on_access_with_source_line(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('not json\n')
    ds = JsonlTextDataset([path], TinyTokenizer(), 8, target_samples=1)
    with pytest.raises(ValueError, match=r'bad.jsonl:1'):
        ds[0]


def test_teacher_accepts_raw_prompt_without_decode(monkeypatch):
    client = OnlineTeacherClient("http://teacher", "Qwen", "/unused", top_k=1, cache_size=0)

    class Tokenizer(TinyTokenizer):
        def decode(self, ids, **kwargs):
            raise AssertionError("raw JSONL prompt must bypass decode")

    client._get_tokenizer = lambda: Tokenizer()

    class Response:
        status_code = 200
        headers = {}
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"prompt_logprobs": [None, {"5": {"logprob": -1.0}}]}]}

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    client._get_session = lambda: Session()
    result = client.get_logits(torch.tensor([[5, 6]]), prompts=["ab"])
    assert result.token_ids[0, 1, 0].item() == 5
