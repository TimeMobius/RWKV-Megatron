from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from rwkv_megatron.checkpoint_retention import prune_checkpoints


def checkpoint(root, iteration, distributed=False):
    folder = root / f'iter_{iteration:07d}'
    folder.mkdir()
    if distributed:
        (folder / '.metadata').write_bytes(b'metadata')
    else:
        rank = folder / 'mp_rank_00'
        rank.mkdir()
        (rank / 'model_optim_rng.pt').write_bytes(b'checkpoint')
    return folder


@pytest.mark.parametrize('distributed', [False, True])
def test_retention_keeps_latest_n_and_ignores_inflight_release_symlinks(tmp_path, distributed):
    folders = [checkpoint(tmp_path, i, distributed) for i in (0, 100, 200, 300, 400)]
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text('300')
    for name in ('release', 'logs', 'iter_0000050'):
        (tmp_path / name).mkdir()
    outside = tmp_path / 'external'
    outside.mkdir()
    (tmp_path / 'iter_0000075').symlink_to(outside, target_is_directory=True)
    removed = prune_checkpoints(tmp_path, 2)
    assert removed == [str(folders[1])]
    assert not folders[1].exists()
    assert all(p.exists() for p in (folders[0], *folders[2:]))
    assert (tmp_path / 'release').is_dir()
    assert (tmp_path / 'logs').is_dir()
    assert (tmp_path / 'iter_0000050').exists()  # No recognizable checkpoint.
    assert (tmp_path / 'iter_0000075').is_symlink() and outside.exists()
    assert prune_checkpoints(tmp_path, 2) == []


@pytest.mark.parametrize('tracker', [None, '', 'release', 'garbage', '999', '0'])
def test_bad_or_incomplete_tracker_never_deletes(tmp_path, tracker):
    paths = [checkpoint(tmp_path, i) for i in (100, 200, 300)]
    if tracker is not None:
        (tmp_path / 'latest_checkpointed_iteration.txt').write_text(tracker)
    assert prune_checkpoints(tmp_path, 1) == []
    assert all(p.exists() for p in paths)


def test_disabled_pruning(tmp_path):
    paths = [checkpoint(tmp_path, i) for i in (100, 200)]
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text('200')
    assert prune_checkpoints(tmp_path, 0) == []
    assert all(p.exists() for p in paths)
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path, -1)


def test_config_validation(tmp_path, monkeypatch):
    import argparse
    from rwkv_megatron.arguments import add_rwkv_arguments
    path = tmp_path / 'config.yaml'
    monkeypatch.setattr(sys, 'argv', ['train', '--rwkv-config', str(path)])
    path.write_text('rwkv_checkpoint_keep_last: 2\n')
    assert add_rwkv_arguments(argparse.ArgumentParser()).parse_args().rwkv_checkpoint_keep_last == 2
    path.write_text('rwkv_checkpoint_keep_last: -1\n')
    with pytest.raises(ValueError, match='keep_last'):
        add_rwkv_arguments(argparse.ArgumentParser())


def test_callback_runs_on_rank_zero_and_on_new_completed_checkpoint(tmp_path):
    from rwkv_megatron.checkpoint_retention import make_retention_callback
    checkpoint(tmp_path, 1)
    checkpoint(tmp_path, 2)
    tracker = tmp_path / 'latest_checkpointed_iteration.txt'
    tracker.write_text('2')
    args = SimpleNamespace(save=str(tmp_path), rwkv_checkpoint_keep_last=1, rank=1)
    callback = make_retention_callback(args)
    callback.prune()
    assert (tmp_path / 'iter_0000001').exists()
    args.rank = 0
    callback.on_train_step_start(None)
    assert not (tmp_path / 'iter_0000001').exists()
    checkpoint(tmp_path, 3)
    tracker.write_text('3')
    callback.on_train_end(None)
    assert not (tmp_path / 'iter_0000002').exists()
    assert (tmp_path / 'iter_0000003').exists()
