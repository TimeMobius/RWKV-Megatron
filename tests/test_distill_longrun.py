"""Reject incomplete checkpoints and failed/gapped restart logs before phase two."""

import pytest

from tools.run_distill_longrun import audit_training_log, checkpoint_iteration


def test_checkpoint_requires_all_pipeline_tensor_shards(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2")
    for rank in ("00_000", "00_001", "01_000"):
        folder = tmp_path / "iter_0000002" / ("mp_rank_" + rank)
        folder.mkdir(parents=True)
        (folder / "model_optim_rng.pt").write_bytes(b"saved")
    with pytest.raises(RuntimeError, match="Incomplete"):
        checkpoint_iteration(tmp_path)
    folder = tmp_path / "iter_0000002/mp_rank_01_001"
    folder.mkdir()
    (folder / "model_optim_rng.pt").touch()
    with pytest.raises(RuntimeError, match="Incomplete"):
        checkpoint_iteration(tmp_path)
    (folder / "model_optim_rng.pt").write_bytes(b"saved")
    assert checkpoint_iteration(tmp_path) == 2


def log_line(step, skipped=0, nan=0, teacher=0, norm="1.0"):
    return (
        f" iteration {step}/100 | consumed samples: {step*8} | teacher/failed: {teacher} | "
        f"grad norm: {norm} | number of skipped iterations: {skipped} | number of nan iterations: {nan} |\n"
    )


def test_resumed_log_requires_loaded_checkpoint_and_next_step(tmp_path):
    path = tmp_path / "phase.log"
    prefix = "successfully loaded checkpoint from /test at iteration 2\n"
    path.write_text(prefix + log_line(3) + log_line(4))
    report = audit_training_log(path, 2)
    assert report["optimizer_steps"] == 2 and report["first_iteration"] == 3
    path.write_text(log_line(3))
    with pytest.raises(RuntimeError, match="checkpoint load"):
        audit_training_log(path, 2)
    path.write_text(prefix + log_line(4))
    with pytest.raises(RuntimeError, match="first iteration"):
        audit_training_log(path, 2)


@pytest.mark.parametrize(
    "bad",
    [
        log_line(2),
        log_line(1, skipped=1),
        log_line(1, nan=1),
        log_line(1, teacher=1),
        log_line(1, norm="nan"),
        log_line(1, norm="inf"),
        log_line(1) + log_line(3),
    ],
)
def test_failed_or_gapped_log_is_rejected(tmp_path, bad):
    path = tmp_path / "phase.log"
    path.write_text(bad)
    with pytest.raises(RuntimeError):
        audit_training_log(path, 0)
