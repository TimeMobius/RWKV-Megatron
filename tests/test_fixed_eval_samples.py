"""Fixed validation restarts real samplers without touching train/test streams."""
import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from megatron.core.rerun_state_machine import RerunMode
from megatron.training.datasets.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from megatron.training.datasets.fixed_eval import FixedEvalDataIterator, reset_fixed_eval_iterator
from megatron.training.training import cyclic_iter
from rwkv_megatron.arguments import add_rwkv_arguments


class SampleIDs(Dataset):
    def __len__(self):
        return 40

    def __getitem__(self, index):
        return index


def loader(rank=0, cyclic=False, workers=0):
    dataset = SampleIDs()
    if cyclic:
        sampler = MegatronPretrainingRandomSampler(dataset, 40, 0, 2, rank, 2, True)
    else:
        sampler = MegatronPretrainingSampler(40, 0, 2, rank, 2)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      persistent_workers=workers > 0, generator=torch.Generator().manual_seed(19))


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('workers', [0, 2])
def test_repeated_validation_replays_samples_and_rank_partition(cyclic, workers):
    iterators = [FixedEvalDataIterator(loader(rank, cyclic, workers), cyclic_iter if cyclic else None)
                 for rank in (0, 1)]
    machine = SimpleNamespace(get_mode=lambda: RerunMode.DISABLED)
    with patch('megatron.core.rerun_state_machine.get_rerun_state_machine', return_value=machine):
        first = []
        reset_fixed_eval_iterator(iterators)
        for iterator in iterators:
            first.append([next(iterator).tolist() for _ in range(3)])
        # Drop a partially consumed eval too, including prefetched batches.
        next(iterators[0])
        for _ in range(3):
            reset_fixed_eval_iterator(iterators)
            actual = [[next(iterator).tolist() for _ in range(3)] for iterator in iterators]
            assert actual == first
        assert set(sum(first[0], [])).isdisjoint(sum(first[1], []))
        for rank, iterator in enumerate(iterators):
            restored = FixedEvalDataIterator(loader(rank, cyclic, workers), cyclic_iter if cyclic else None)
            restored.reset_for_eval()
            assert [next(restored).tolist() for _ in range(3)] == first[rank]


def test_nested_virtual_pipeline_iterators_and_none():
    fixed = FixedEvalDataIterator(loader())
    reset_fixed_eval_iterator([None, [fixed]])
    with patch('megatron.core.rerun_state_machine.get_rerun_state_machine',
               return_value=SimpleNamespace(get_mode=lambda: RerunMode.DISABLED)):
        assert next(fixed).tolist() == [0, 1]
    reset_fixed_eval_iterator(None)
    with pytest.raises(ValueError, match='restartable'):
        reset_fixed_eval_iterator(iter([1, 2]))


def test_restart_does_not_consume_global_random_state():
    fixed = FixedEvalDataIterator(loader())
    before = torch.random.get_rng_state().clone()
    for _ in range(3):
        fixed.reset_for_eval()
    assert torch.equal(before, torch.random.get_rng_state())


def test_yaml_boolean_default_and_cli_override(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train'])
    parser = add_rwkv_arguments(argparse.ArgumentParser())
    assert parser.parse_args([]).fixed_eval_samples is False
    path = tmp_path/'config.yaml'
    path.write_text('fixed_eval_samples: true\n')
    monkeypatch.setattr(sys, 'argv', ['train', '--rwkv-config', str(path)])
    parser = add_rwkv_arguments(argparse.ArgumentParser())
    assert parser.parse_args().fixed_eval_samples is True
    assert parser.parse_args(['--no-fixed-eval-samples']).fixed_eval_samples is False
    path.write_text('fixed_eval_samples: "true"\n')
    with pytest.raises(ValueError, match='YAML boolean'):
        add_rwkv_arguments(argparse.ArgumentParser())
