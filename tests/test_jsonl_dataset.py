import json
from pathlib import Path
from types import SimpleNamespace

import torch

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
