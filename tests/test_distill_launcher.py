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

from tools.launch_distillation import default_training_command, launch


@pytest.fixture
def programs(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(rwkv_distill=True, rwkv_teacher_pool_port=port)))
    pool = tmp_path / 'pool.py'
    pool.write_text('''import json, os, socket, sys, time
from pathlib import Path
port, pidfile, mode = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
pidfile.write_text(str(os.getpid()))
if mode == 'fail': sys.exit(7)
sock=socket.socket(); sock.bind(('127.0.0.1',port)); sock.listen()
if mode != 'timeout': print(json.dumps({'event':'teacher_pool_ready'}),flush=True)
if mode == 'crash': time.sleep(.3); sys.exit(9)
while True: time.sleep(.1)
''')
    train = tmp_path / 'train.py'
    train.write_text('''import os, socket, sys, time
from pathlib import Path
port, pidfile, code = int(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
assert '--rwkv-teacher-pool' in sys.argv
with socket.create_connection(('127.0.0.1',port),timeout=1): pass
pidfile.write_text(str(os.getpid()))
if code == 99:
    while True: time.sleep(.1)
sys.exit(code)
''')
    return config, pool, train, port, tmp_path


def is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize('code', [0, 3])
def test_train_exit_status_and_pool_cleanup(programs, code):
    config, pool, train, port, folder = programs
    pool_pid, train_pid = folder / 'pool.pid', folder / 'train.pid'
    result = launch(config, [sys.executable, str(train), str(port), str(train_pid), str(code)],
                    pool_command=[sys.executable, '-u', str(pool), str(port), str(pool_pid), 'ok'])
    assert result == code
    assert not is_alive(int(pool_pid.read_text()))
    assert not is_alive(int(train_pid.read_text()))


@pytest.mark.parametrize('mode,exception', [('fail', RuntimeError), ('timeout', TimeoutError)])
def test_startup_failure_never_starts_training(programs, mode, exception):
    config, pool, train, port, folder = programs
    pool_pid, train_pid = folder / 'pool.pid', folder / 'train.pid'
    with pytest.raises(exception):
        launch(config, [sys.executable, str(train), str(port), str(train_pid), '0'], startup_timeout=.4,
               pool_command=[sys.executable, '-u', str(pool), str(port), str(pool_pid), mode])
    assert not train_pid.exists()
    assert not is_alive(int(pool_pid.read_text()))


def test_pool_crash_stops_training(programs):
    config, pool, train, port, folder = programs
    pool_pid, train_pid = folder / 'pool.pid', folder / 'train.pid'
    with pytest.raises(RuntimeError, match='during training'):
        launch(config, [sys.executable, str(train), str(port), str(train_pid), '99'],
               pool_command=[sys.executable, '-u', str(pool), str(port), str(pool_pid), 'crash'])
    assert not is_alive(int(pool_pid.read_text()))
    assert not is_alive(int(train_pid.read_text()))


def test_sigterm_cleans_up_pool_and_training(programs):
    config, pool, train, port, folder = programs
    pool_pid, train_pid = folder / 'pool.pid', folder / 'train.pid'
    wrapper = folder / 'wrapper.py'
    wrapper.write_text('from tools.launch_distillation import default_training_command, launch\nimport sys\nsys.exit(launch(' +
                       repr(str(config)) + ', ' + repr([sys.executable, str(train), str(port), str(train_pid), '99']) +
                       ', pool_command=' + repr([sys.executable, '-u', str(pool), str(port), str(pool_pid), 'ok']) + '))\n')
    process = subprocess.Popen([sys.executable, str(wrapper)], env={**os.environ, 'PYTHONPATH': str(Path.cwd())})
    try:
        deadline = time.monotonic() + 10
        while not train_pid.exists():
            assert time.monotonic() < deadline and process.poll() is None
            time.sleep(.05)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 143
        assert not is_alive(int(train_pid.read_text()))
        assert not is_alive(int(pool_pid.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


@pytest.mark.parametrize('tp,pp', [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2)])
def test_default_launcher_matches_parallel_layout(tp, pp):
    command = default_training_command({'model_parallel': {
        'tensor_model_parallel_size': tp, 'pipeline_model_parallel_size': pp}})
    assert f'--nproc-per-node={tp * pp}' in command


def test_default_launcher_rejects_cp():
    with pytest.raises(ValueError, match='context parallelism'):
        default_training_command({'model_parallel': {'context_parallel_size': 2}})
