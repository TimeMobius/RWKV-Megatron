"""Subprocess lifecycle tests: no GPU or external teacher required."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tools.launch_distillation import (
    default_training_command,
    launch,
    validate_training_command,
)


@pytest.fixture
def programs(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(dict(rwkv_distill=True, rwkv_teacher_pool_port=port))
    )
    pool = tmp_path / "pool.py"
    pool.write_text("""import json, os, socket, sys, time
from pathlib import Path
port, pidfile, mode = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
pidfile.write_text(str(os.getpid()))
if mode == 'fail': sys.exit(7)
sock=socket.socket(); sock.bind(('127.0.0.1',port)); sock.listen()
if mode != 'timeout': print(json.dumps({'event':'teacher_pool_ready'}),flush=True)
if mode == 'crash': time.sleep(.3); sys.exit(9)
while True: time.sleep(.1)
""")
    train = tmp_path / "train.py"
    train.write_text("""import os, socket, sys, time
from pathlib import Path
port, pidfile, code = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
assert '--rwkv-teacher-pool' in sys.argv
with socket.create_connection(('127.0.0.1',port),timeout=1): pass
pidfile.write_text(str(os.getpid()))
if code == 99:
    while True: time.sleep(.1)
sys.exit(code)
""")
    return config, pool, train, port, tmp_path


def is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize("code", [0, 3])
def test_train_exit_status_and_pool_cleanup(programs, code):
    config, pool, train, port, folder = programs
    pool_pid, train_pid = folder / "pool.pid", folder / "train.pid"
    result = launch(
        config,
        [sys.executable, str(train), str(port), str(train_pid), str(code)],
        pool_command=[sys.executable, "-u", str(pool), str(port), str(pool_pid), "ok"],
    )
    assert result == code
    assert not is_alive(int(pool_pid.read_text()))
    assert not is_alive(int(train_pid.read_text()))
    logs = list(folder.glob('teacher-pool-*.log'))
    assert len(logs) == 1
    assert 'teacher_pool_ready' in logs[0].read_text()


@pytest.mark.parametrize(
    "mode,exception", [("fail", RuntimeError), ("timeout", TimeoutError)]
)
def test_startup_failure_never_starts_training(programs, mode, exception):
    config, pool, train, port, folder = programs
    pool_pid, train_pid = folder / "pool.pid", folder / "train.pid"
    with pytest.raises(exception):
        launch(
            config,
            [sys.executable, str(train), str(port), str(train_pid), "0"],
            startup_timeout=0.4,
            pool_command=[
                sys.executable,
                "-u",
                str(pool),
                str(port),
                str(pool_pid),
                mode,
            ],
        )
    assert not train_pid.exists()
    assert not is_alive(int(pool_pid.read_text()))


def test_pool_crash_stops_training(programs):
    config, pool, train, port, folder = programs
    pool_pid, train_pid = folder / "pool.pid", folder / "train.pid"
    with pytest.raises(RuntimeError, match="during training"):
        launch(
            config,
            [sys.executable, str(train), str(port), str(train_pid), "99"],
            pool_command=[
                sys.executable,
                "-u",
                str(pool),
                str(port),
                str(pool_pid),
                "crash",
            ],
        )
    assert not is_alive(int(pool_pid.read_text()))
    assert not is_alive(int(train_pid.read_text()))


def test_sigterm_cleans_up_pool_and_training(programs):
    config, pool, train, port, folder = programs
    pool_pid, train_pid = folder / "pool.pid", folder / "train.pid"
    wrapper = folder / "wrapper.py"
    wrapper.write_text(
        "from tools.launch_distillation import default_training_command, launch\nimport sys\nsys.exit(launch("
        + repr(str(config))
        + ", "
        + repr([sys.executable, str(train), str(port), str(train_pid), "99"])
        + ", pool_command="
        + repr([sys.executable, "-u", str(pool), str(port), str(pool_pid), "ok"])
        + "))\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(wrapper)],
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    try:
        deadline = time.monotonic() + 10
        while not train_pid.exists():
            assert time.monotonic() < deadline and process.poll() is None
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 143
        assert not is_alive(int(train_pid.read_text()))
        assert not is_alive(int(pool_pid.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2)])
def test_default_launcher_matches_parallel_layout(tp, pp):
    command = default_training_command(
        {
            "model_parallel": {
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
            }
        }
    )
    assert f"--nproc-per-node={tp * pp}" in command


@pytest.mark.parametrize(
    "dp,tp,pp,sp,cp", [(1, 1, 1, False, 2), (2, 2, 1, True, 2), (1, 2, 2, True, 2)]
)
def test_default_launcher_infers_cp(dp, tp, pp, sp, cp, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    cfg = {
        "model_parallel": {
            "data_parallel_size": dp,
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "context_parallel_size": cp,
            "sequence_parallel": sp,
        },
        "seq_length": 128,
        "micro_batch_size": 2,
        "global_batch_size": 8,
    }
    command = default_training_command(cfg)
    assert f"--nproc-per-node={dp*tp*pp*cp}" in command
    validate_training_command(cfg, command)
    with pytest.raises(ValueError, match="disagrees|divisible"):
        validate_training_command(cfg, ["torchrun", "--nproc-per-node=1", "train.py"])
    cfg["seq_length"] = 96
    with pytest.raises(ValueError, match="seq_length"):
        default_training_command(cfg)


@pytest.mark.parametrize("dp,tp,pp", [(2, 1, 1), (2, 2, 2), (4, 2, 1)])
def test_launcher_infers_data_parallel_processes(dp, tp, pp):
    cfg = {
        "model_parallel": {
            "data_parallel_size": dp,
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
        },
        "global_batch_size": 16,
        "micro_batch_size": 2,
    }
    assert f"--nproc-per-node={dp * tp * pp}" in default_training_command(cfg)


@pytest.mark.parametrize(
    "config,match",
    [
        ({"model_parallel": {"data_parallel_size": True}}, "positive integers"),
        ({"model_parallel": {"data_parallel_size": 0}}, "positive integers"),
        ({"model_parallel": {"sequence_parallel": True}}, "requires"),
        ({"model_parallel": {"sequence_parallel": "true"}}, "boolean"),
        (
            {"model_parallel": {"virtual_pipeline_model_parallel_size": 2}},
            "non-interleaved",
        ),
        (
            {
                "model_parallel": {"tensor_model_parallel_size": 4},
                "language_model": {"num_attention_heads": 6},
            },
            "num_attention_heads",
        ),
        (
            {
                "model_parallel": {"pipeline_model_parallel_size": 2},
                "language_model": {"num_layers": 3},
            },
            "num_layers",
        ),
        (
            {
                "model_parallel": {
                    "tensor_model_parallel_size": 2,
                    "sequence_parallel": True,
                },
                "seq_length": 127,
            },
            "seq_length",
        ),
        (
            {
                "model_parallel": {"data_parallel_size": 4},
                "micro_batch_size": 2,
                "global_batch_size": 12,
            },
            "global_batch_size",
        ),
    ],
)
def test_launcher_rejects_invalid_layouts(config, match):
    with pytest.raises(ValueError, match=match):
        default_training_command(config)


def test_explicit_torchrun_checks_layout_and_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    cfg = {
        "model_parallel": {
            "data_parallel_size": 2,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 2,
        }
    }
    command = default_training_command(cfg)
    validate_training_command(cfg, command)
    with pytest.raises(ValueError, match="disagrees"):
        validate_training_command(cfg, ["torchrun", "--nproc-per-node=4", "train.py"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(ValueError, match="exposes 4"):
        validate_training_command(cfg, command)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0,1,2,3,4,5,6")
    with pytest.raises(ValueError, match="duplicate"):
        validate_training_command(cfg, command)


def test_explicit_torchrun_preserves_implicit_dp_override(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    cfg = {
        "model_parallel": {"tensor_model_parallel_size": 2},
        "micro_batch_size": 2,
        "global_batch_size": 8,
    }
    validate_training_command(cfg, ["torchrun", "--nproc-per-node", "4", "train.py"])
    cfg["global_batch_size"] = 6
    with pytest.raises(ValueError, match="global_batch_size"):
        validate_training_command(cfg, ["torchrun", "--nproc-per-node=4", "train.py"])


@pytest.mark.parametrize("name", ["tp2-sp-pp2-cp2", "tp2-sp-cp4"])
def test_cp_smoke_configs_use_eight_gpus_and_full_context_teacher(name):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (root / "configs" / f"config-qwen-2p7b-distill-{name}-smoke.yaml").read_text()
    )
    assert "--nproc-per-node=8" in default_training_command(cfg)
    assert cfg["seq_length"] == 4096
    assert cfg["rwkv_teacher_pool"] and cfg["rwkv_teacher_replace_bad_samples"]
    assert cfg["rwkv_teacher_tokenizer_use_fast"] is False
    assert cfg["rwkv_teacher_step_timeout"] + 60 < cfg["distributed_timeout_minutes"] * 60
    assert cfg["rwkv_teacher_sample_records"].endswith(f"2p7b-{name}-smoke/sample-selection")


@pytest.mark.parametrize("name", ["tp4-sp-pp2", "tp2-sp-pp2-cp2-offload50", "tp2-sp-pp4-balanced"])
def test_28b_configs_preserve_checkpoint_vocabulary_and_eight_gpu_geometry(name):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / f"config-qwen-28b-distill-{name}-smoke.yaml").read_text())
    assert "--nproc-per-node=8" in default_training_command(cfg)
    multiple = cfg["make_vocab_size_divisible_by"] * cfg["model_parallel"]["tensor_model_parallel_size"]
    assert ((cfg["rwkv_distill_valid_vocab_size"] + multiple - 1) // multiple) * multiple == 248320
    assert cfg["language_model"]["num_layers"] == 88 and cfg["language_model"]["hidden_size"] == 5120
    assert cfg["seq_length"] == 4096 and cfg["use_distributed_optimizer"]
    assert cfg["rwkv_teacher_model"] == "Qwen3.8-27B"
    assert cfg["optimizer_cpu_offload"] == ("offload50" in name)
    if name.endswith("balanced"):
        assert cfg["language_model"]["num_layers_in_first_pipeline_stage"] == 18
        assert cfg["language_model"]["num_layers_in_last_pipeline_stage"] == 18
