"""Quality reports use frozen tokens, effective vocab, and token-weighted CE."""

import json
import math
from types import SimpleNamespace

import pytest
import torch
from tools.evaluate_distill_checkpoint import aggregate, compare, fixed_manifest, score_logits


class Tokenizer:
    eos_token_id = 0
    def get_vocab(self):
        return {'pad': 0, 'a': 1, 'b': 2, 'c': 3}
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) % 3 + 1 for c in text])


def test_manifest_freezes_inputs_and_rejects_source_tokenizer_or_payload_changes(tmp_path):
    path = tmp_path / 'validation.jsonl'
    path.write_text('{"text":"abcdef"}\n{"text":"ghijkl"}\n')
    settings = dict(data_path=[str(path)], seq_length=8, rwkv_teacher_tokenizer_use_fast=False)
    manifest_path = tmp_path / 'manifest.json'
    first = fixed_manifest(manifest_path, settings, Tokenizer(), 2)
    second = fixed_manifest(manifest_path, settings, Tokenizer(), 2)
    assert first == second
    with pytest.raises(ValueError, match='manifest changed'):
        fixed_manifest(manifest_path, dict(settings, rwkv_teacher_tokenizer_use_fast=True), Tokenizer(), 2)
    first['rows'][0]['labels'][0] = (first['rows'][0]['labels'][0] + 1) % 4
    manifest_path.write_text(json.dumps(first))
    with pytest.raises(ValueError, match='manifest changed'):
        fixed_manifest(manifest_path, settings, Tokenizer(), 2)
    manifest_path.unlink()
    fixed_manifest(manifest_path, settings, Tokenizer(), 2)
    path.write_text('{"text":"changed"}\n')
    with pytest.raises(ValueError, match='manifest changed'):
        fixed_manifest(manifest_path, settings, Tokenizer(), 2)


def test_quality_excludes_padding_and_padded_vocab_and_weights_by_tokens():
    logits = torch.tensor([[2., 0., 100.], [0., 2., 100.], [0., 0., 100.]])
    labels = torch.tensor([0, 1, -100])
    row = score_logits(logits, labels, torch.tensor([1., 1., 0.]), 2)
    assert row['tokens'] == row['correct'] == 2
    expected = torch.nn.functional.cross_entropy(logits[:2, :2], labels[:2])
    assert row['nll'] / 2 == pytest.approx(float(expected))
    out = aggregate([row, dict(nll=8., tokens=4, correct=1)])
    assert out['ce'] == pytest.approx((row['nll'] + 8) / 6)
    assert out['ppl'] == pytest.approx(math.exp(out['ce']))
    assert score_logits(logits, labels, torch.zeros(3), 2)['tokens'] == 0


def test_comparison_refuses_different_samples():
    with pytest.raises(ValueError, match='different fixed samples'):
        compare(dict(manifest_sha256='a'), dict(manifest_sha256='b'))
