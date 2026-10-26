"""Deterministic JSONL tokenizer/teacher for real distributed pool training tests."""

import json
from types import SimpleNamespace

import torch

from rwkv_megatron.online_teacher import TeacherSampleError, TeacherTopK


class FixtureTokenizer:
    def get_vocab(self):
        return {chr(0x400 + i): i for i in range(119)}

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) - 0x400 for char in text])


class FixturePoolTeacher:
    prompt_mode = "text"
    server_url = "http://fixture"
    model_name = "fixture"
    top_k = 4
    request_seed = 1234
    request_temperature = 1.0

    def _get_tokenizer(self):
        return FixtureTokenizer()

    def get_effective_vocab_size(self):
        return 119

    def get_logits(self, tokens, **kwargs):
        if (tokens[:, 0] % 7 == 0).any():
            raise TeacherSampleError("injected sample token alignment mismatch")
        ids = (tokens[..., None] + torch.tensor([0, 31, 67, 98])) % 119
        lp = -torch.tensor([0.2, 1.1, 2.3, 4.0]).expand_as(ids).clone()
        ids = ids.masked_fill((tokens % 23 == 0)[..., None], -1)
        return TeacherTopK(ids, lp)

    def get_last_request_metrics(self):
        return dict(latency_seconds=0.01)

    def get_stats(self):
        return {}


def write_fixture(path, split):
    with path.open("w") as stream:
        for index in range(128):
            generator = torch.Generator().manual_seed(20261023 + split * 100000 + index)
            tokens = torch.randint(1, 119, (129 - (index % 5) * 9,), generator=generator)
            text = "".join(chr(0x400 + int(token)) for token in tokens)
            stream.write(json.dumps(dict(text=text)) + "\n")
