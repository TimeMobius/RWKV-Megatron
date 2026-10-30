"""A finite epoch never trains on repeated text, including rejected tail rows."""

import argparse
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from rwkv_megatron.arguments import add_rwkv_arguments
from rwkv_megatron.jsonl_dataset import JsonlTextDataset
from rwkv_megatron.online_teacher import TeacherSampleError, TeacherTopK
from rwkv_megatron.teacher_pool import TeacherPool, _LocalPoolRequestClient
from rwkv_megatron.teacher_replacement import selection_signature
from rwkv_megatron.teacher_sample_stream import PoolValidatedDataset


class Tokenizer:
    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(text.encode()))


class Teacher:
    server_url = 'http://fixture'
    model_name = 'fixture'
    top_k = 2
    request_seed = 1234
    request_temperature = 1.0
    prompt_mode = 'text'

    def __init__(self):
        self.calls = []

    def _get_tokenizer(self):
        return Tokenizer()

    def get_effective_vocab_size(self):
        return 128

    def get_logits(self, tokens, prompts=None, **kwargs):
        self.calls.extend(prompts)
        if prompts == ['bb']:
            raise TeacherSampleError('bad tail')
        return TeacherTopK(tokens[..., None].expand(-1, -1, 2).int(),
                           torch.zeros(*tokens.shape, 2))

    def get_stats(self):
        return {}

    def get_last_request_metrics(self):
        return dict(latency_seconds=0.01)


def test_finite_epoch_masks_only_tail_padding_and_preserves_default_repetition(tmp_path):
    path = tmp_path/'train.jsonl'
    path.write_text(''.join(json.dumps(dict(text=s))+'\n' for s in ['abc', 'def', 'ghi']))
    finite = JsonlTextDataset([path], Tokenizer(), 4, target_samples=4, epoch_windows=3)
    legacy = JsonlTextDataset([path], Tokenizer(), 4, target_samples=4)
    for i in range(3):
        assert finite[i]['raw_prompts'] == legacy[i]['raw_prompts']
        assert torch.equal(finite[i]['loss_mask'], legacy[i]['loss_mask'])
    assert finite[3]['epoch_padding'] and finite[3]['loss_mask'].sum()==0
    assert legacy[3]['loss_mask'].sum()==2
    assert finite[0]['loss_mask'].sum()==2
    assert finite.get_source_sample(100)['loss_mask'].sum()==0


def test_separate_pool_uses_epoch_boundary_and_never_requests_padding(tmp_path):
    path = tmp_path/'train.jsonl'
    path.write_text(''.join(json.dumps(dict(text=s))+'\n' for s in ['aa','cc','bb']))
    source = JsonlTextDataset([path], Tokenizer(), 4, target_samples=4, epoch_windows=3)
    settings = SimpleNamespace(rwkv_jsonl_max_records=0, rwkv_teacher_model='fixture',
        rwkv_teacher_url='http://fixture', rwkv_teacher_top_k=2,
        rwkv_teacher_seed=1234, rwkv_teacher_prompt_mode='text')
    teacher=Teacher()
    pool=TeacherPool(teacher,workers=2,max_results=8)
    reader=PoolValidatedDataset(source,_LocalPoolRequestClient(pool),tmp_path/'journal',
        selection_signature(source,settings,'train'),lookahead=2,delivery=None)
    try:
        rows=[reader[i] for i in range(4)]
        assert [r['raw_prompts'] for r in rows[:2]]==['aa','cc']
        assert [r['loss_mask'].sum() for r in rows]==[1,1,0,0]
        assert all(r['teacher_token_ids'].shape==(4,2) for r in rows)
        assert sorted(teacher.calls)==['aa','bb','cc']
        assert reader.rejected_count()==1
    finally:
        reader.close()
        pool.close()


@pytest.mark.parametrize('value',[-1,True,1.5])
def test_epoch_bound_requires_nonnegative_integer(tmp_path,monkeypatch,value):
    config=tmp_path/'config.yaml'
    config.write_text('rwkv_jsonl_epoch_windows: '+str(value).lower()+'\n')
    monkeypatch.setattr(sys,'argv',['train','--rwkv-config',str(config)])
    with pytest.raises(ValueError,match='epoch_windows'):
        add_rwkv_arguments(argparse.ArgumentParser())
