"""Retain the most recent completed Megatron checkpoints without core changes."""

from pathlib import Path
import re
import shutil


_ITERATION = re.compile(r'iter_([0-9]{7,})\Z')


def _is_checkpoint(path):
    if path.is_symlink() or not path.is_dir():
        return False
    # Recognize both legacy torch rank directories and distributed formats.
    return ((path / '.metadata').is_file() or (path / 'metadata.json').is_file()
            or any(path.glob('mp_rank_*/model_optim_rng.pt')))


def prune_checkpoints(save_dir, keep_last):
    """Delete only completed ordinary iterations at/below the durable tracker.

    An absent/invalid tracker or missing latest checkpoint disables deletion.
    Newer directories may still be in flight; release, symlinks, initialization
    checkpoints and unrelated directories are never removed. Call after save
    finalization or from the next training step, on global rank zero only.
    """
    if type(keep_last) is not int or keep_last < 0:
        raise ValueError('keep_last must be a nonnegative integer')
    if not keep_last or not save_dir:
        return []
    root = Path(save_dir)
    tracker = root / 'latest_checkpointed_iteration.txt'
    try:
        latest = int(tracker.read_text().strip())
    except (FileNotFoundError, ValueError):
        return []
    latest_dir = root / f'iter_{latest:07d}'
    if latest < 1 or not _is_checkpoint(latest_dir):
        return []
    candidates = []
    for path in root.iterdir():
        match = _ITERATION.fullmatch(path.name)
        if match and 0 < int(match[1]) <= latest and _is_checkpoint(path):
            candidates.append((int(match[1]), path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    removed = []
    for iteration, path in candidates[keep_last:]:
        if path.is_symlink():
            continue
        shutil.rmtree(path)
        removed.append(str(path))
    return removed


def make_retention_callback(args):
    """Keep Megatron imports out of the filesystem-only helper and its tests."""
    from megatron.training.callbacks import Callback

    if args.rwkv_checkpoint_keep_last and getattr(args, 'save_retain_interval', None) is not None:
        raise ValueError('rwkv_checkpoint_keep_last cannot be combined with save_retain_interval')

    class CheckpointRetentionCallback(Callback):
        def __init__(self):
            self.last_tracker = None

        def prune(self):
            if not args.rwkv_checkpoint_keep_last or not args.save or args.rank != 0:
                return
            tracker = Path(args.save) / 'latest_checkpointed_iteration.txt'
            try:
                current = tracker.read_text()
            except FileNotFoundError:
                return
            if current == self.last_tracker:
                return
            removed = prune_checkpoints(args.save, args.rwkv_checkpoint_keep_last)
            self.last_tracker = current
            if removed:
                print(f'checkpoint retention removed: {removed}', flush=True)

        def on_train_step_start(self, context):
            self.prune()

        def on_train_end(self, context):
            self.prune()

    return CheckpointRetentionCallback()
