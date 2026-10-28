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


def test_pool_audit_allows_replaced_samples_but_rejects_fatal_or_missing_metrics(tmp_path):
    path = tmp_path / 'pool.log'
    def pool_line(fatal=0, sample=2):
        return log_line(1).replace('teacher/failed: 0',
            f'teacher/pool_failed: {fatal+sample} | teacher/pool_sample_errors: {sample} | teacher/pool_fatal_errors: {fatal}')
    path.write_text(pool_line())
    report = audit_training_log(path, 0)
    assert report['teacher_failures'] == 0
    assert report['teacher_sample_errors'] == 2
    assert report['teacher_failure_metrics'] == ['teacher/pool_fatal_errors']
    path.write_text(pool_line(fatal=1))
    with pytest.raises(RuntimeError, match='Teacher failures'):
        audit_training_log(path, 0)
    path.write_text(log_line(1).replace('teacher/failed: 0 | ', ''))
    with pytest.raises(RuntimeError, match='Missing teacher/failed metric'):
        audit_training_log(path, 0)


def test_pool_audit_rejects_nonfinite_failure_counters(tmp_path):
    path = tmp_path / 'pool.log'
    path.write_text(log_line(1).replace('teacher/failed: 0', 'teacher/pool_fatal_errors: nan'))
    with pytest.raises(RuntimeError, match='Non-finite'):
        audit_training_log(path, 0)


@pytest.mark.parametrize('capacity', [False, True])
@pytest.mark.parametrize('quality_exit', [0, 1])
@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.parametrize('cp', [1, 2])
def test_supervisor_evaluates_the_final_checkpoint_and_propagates_evaluation_failure(tmp_path, monkeypatch, quality_exit, continuous, capacity, cp):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import yaml
    import tools.run_distill_longrun as runner
    import tools.check_teacher_tokenizer as tokenizer_check
    import tools.evaluate_distill_checkpoint as evaluator
    import wandb

    root = tmp_path / 'repo'
    (root / 'configs').mkdir(parents=True)
    (root / 'configs/config-qwen-2p7b-distill-tp2-sp-pp2-smoke.yaml').write_text(
        yaml.safe_dump(dict(model_parallel={}, language_model={}, load_pretrained_rwkv='initial.pt', rwkv_teacher_tokenizer='fixture')))
    output = tmp_path / 'run'
    commands = []
    class Process:
        pid = 999999
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.quality = '--manifest' in command
            if self.quality:
                assert command[command.index('--checkpoint')+1].endswith('iter_0000002' if continuous else 'iter_0000003')
                assert '--nproc-per-node=1' in command
                report = Path(command[command.index('--output')+1])
                report.write_text('{}')
            else:
                config = yaml.safe_load(Path(command[-1]).read_text())
                assert config['model_parallel']['context_parallel_size'] == cp
                assert config['model_parallel']['data_parallel_size'] == 2 // cp
                assert (config['model_parallel']['tensor_model_parallel_size']
                        * config['model_parallel']['pipeline_model_parallel_size']
                        * cp * config['model_parallel']['data_parallel_size']) == 8
                assert config['seq_length'] == (4096 if capacity else 1024)
                assert config['micro_batch_size'] == config['eval_micro_batch_size'] == (2 if capacity else 1)
                if capacity:
                    assert config['language_model']['recompute_granularity'] is None
                if continuous:
                    assert config['exit_duration_in_mins'] == 720
                    assert config['exit_interval'] is None and config['save_interval'] == (1000 if cp == 2 else 250)
                    assert config['rwkv_teacher_prefetch_lookahead'] == 32
                    assert config['rwkv_teacher_pool_workers'] == (4 if capacity else 8)
                    assert config['rwkv_teacher_prefetch_min_workers'] == (2 if capacity else 4)
                iteration = 2 if len(commands) == 1 else 3
                checkpoints = Path(config['save'])
                checkpoints.mkdir(exist_ok=True)
                (checkpoints / 'latest_checkpointed_iteration.txt').write_text(str(iteration))
                for rank in ('00_000','00_001','01_000','01_001'):
                    directory = checkpoints / f'iter_{iteration:07d}' / f'mp_rank_{rank}'
                    directory.mkdir(parents=True)
                    (directory / 'model_optim_rng.pt').write_bytes(b'fixture')
                text = (log_line(1)+log_line(2) if iteration == 2 else
                        'successfully loaded checkpoint from /fixture at iteration 2\n'+log_line(3))
                kwargs['stdout'].write(text)
        def wait(self):
            return quality_exit if self.quality else 0
        def poll(self):
            return self.wait()
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner.sys, 'argv', ['run', '--output', str(output)] +
                        (['--context-parallel-size', '2', '--save-interval', '1000'] if cp == 2 else []) +
                        (['--continuous-minutes', '720'] if continuous else ['--smoke']) +
                        (['--seq-length', '4096', '--micro-batch-size', '2', '--no-recompute',
                          '--teacher-pool-workers', '4'] if capacity else []))
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'test-commit\n')
    monkeypatch.setattr(runner.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(tokenizer_check, 'check_teacher_tokenizer', lambda settings: dict(all_matched=True))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(evaluator, 'fixed_manifest', lambda *a, **k: dict(sha256='fixed-before-training'))
    monkeypatch.setattr(wandb, 'Api', lambda: SimpleNamespace(run=lambda *a: SimpleNamespace(state='finished', url='training-url')))
    monkeypatch.setattr(evaluator, 'upload_quality_report', lambda path, cfg: dict(report=str(path), wandb_url='quality-url'))
    if quality_exit:
        with pytest.raises(RuntimeError, match='quality evaluation exited 1'):
            runner.main()
    else:
        runner.main()
    state = json.loads((output / 'status.json').read_text())
    assert len(commands) == (2 if continuous else 3)
    assert state['state'] == ('failed' if quality_exit else 'completed')
    assert len(state['phases']) == (1 if continuous else 2)
    assert state['planned_restart'] is (not continuous)
    assert state['layout'] == dict(dp=2 // cp, tp=2, pp=2, cp=cp, sp=True)
    assert state['fixed_validation_sha256'] == 'fixed-before-training'
    if not quality_exit:
        assert state['checkpoint_quality']['wandb_url'] == 'quality-url'
