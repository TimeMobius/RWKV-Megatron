"""Tokenizer preflight uses the selected implementation and refuses bad probes."""

from types import SimpleNamespace

import pytest
import torch

from tools.check_teacher_tokenizer import PROBES, check_teacher_tokenizer


class ProbeClient:
    def __init__(self, **settings):
        self.settings = settings
        self.tokenizer_use_fast = settings["tokenizer_use_fast"]
        self.calls = []

    def _get_tokenizer(self):
        return lambda text, **kwargs: SimpleNamespace(
            input_ids=[ord(char) for char in text]
        )

    def get_logits(self, tokens, prompts):
        self.calls.append(prompts[0])
        assert tokens[0].tolist() == [ord(char) for char in prompts[0]]
        return SimpleNamespace(token_ids=torch.empty(*tokens.shape, 1))


def test_preflight_checks_multilingual_text_with_configured_backend(monkeypatch):
    clients = []

    def create(**settings):
        client = ProbeClient(**settings)
        clients.append(client)
        return client

    monkeypatch.setattr("tools.check_teacher_tokenizer.OnlineTeacherClient", create)
    report = check_teacher_tokenizer(
        {"tokenizer_model": "/unused", "rwkv_teacher_tokenizer_use_fast": False}
    )
    assert clients[0].calls == list(PROBES)
    assert clients[0].settings["tokenizer_use_fast"] is False
    assert report["all_matched"] and report["optimizer_steps"] == 0


def test_mismatch_stops_preflight_without_claiming_success(monkeypatch):
    class MismatchClient(ProbeClient):
        def get_logits(self, tokens, prompts):
            raise RuntimeError("Teacher returned incompatible prompt positions")

    monkeypatch.setattr(
        "tools.check_teacher_tokenizer.OnlineTeacherClient", MismatchClient
    )
    with pytest.raises(RuntimeError, match="incompatible"):
        check_teacher_tokenizer({"tokenizer_model": "/unused"})
