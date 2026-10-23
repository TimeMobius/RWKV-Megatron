"""Launch a node-local teacher pool and training, and clean up owned processes.

Run in the activated training environment. vLLM stays an external service.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def stop_process_group(process, timeout=15):
    """Stop the entire owned group, including children of torchrun or the pool."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # A parent can exit before its descendants: still clean up its process group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def wait_for_pool(process, log, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'teacher pool exited during startup (code {process.returncode})')
        line = log.readline()
        if line:
            print(line, end='', flush=True)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('event') == 'teacher_pool_ready':
                return event
        else:
            time.sleep(0.05)
    raise TimeoutError('teacher pool startup timed out')


def default_training_command(settings):
    """One node, DP=1 by default; explicit commands can select more DP ranks."""
    parallel = settings.get('model_parallel', {})
    tp = parallel.get('tensor_model_parallel_size', 1)
    pp = parallel.get('pipeline_model_parallel_size', 1)
    cp = parallel.get('context_parallel_size', 1)
    if any(type(size) is not int or size < 1 for size in (tp, pp, cp)):
        raise ValueError('parallel sizes must be positive integers')
    if cp != 1:
        raise ValueError('online distillation does not support context parallelism')
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc-per-node={tp * pp}', str(ROOT / 'pretrain_rwkv7.py')]


def launch(config, command=None, startup_timeout=120, shutdown_timeout=15, *, pool_command=None):
    if startup_timeout <= 0 or shutdown_timeout <= 0:
        raise ValueError('startup and shutdown timeouts must be positive')
    config = Path(config).resolve()
    settings = yaml.safe_load(config.read_text())
    if not settings.get('rwkv_distill', False):
        raise ValueError('launcher requires rwkv_distill: true')
    host = settings.get('rwkv_teacher_pool_host', '127.0.0.1')
    port = settings.get('rwkv_teacher_pool_port', 50055)
    # Refuse an existing listener so a failed new pool never attaches training
    # to an unrelated server. Only processes started here are terminated.
    try:
        with socket.create_connection((host, port), timeout=0.5):
            raise RuntimeError(f'teacher pool address is already in use: {host}:{port}')
    except (ConnectionRefusedError, TimeoutError):
        pass
    pool_command = pool_command or [sys.executable, '-u', str(ROOT / 'tools/teacher_pool_server.py'),
                                     '--rwkv-config', str(config)]
    command = command or default_training_command(settings)
    train_command = [*command, '--rwkv-config', str(config), '--rwkv-teacher-pool']
    pool = train = None
    old_handlers = {}
    def interrupt(signum, frame):
        raise Interrupted(signum)
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupt)
    try:
        # Separate reader/writer handles avoid shared seek offsets between the
        # subprocess and the parent. Keep the log to report startup failures.
        with tempfile.TemporaryDirectory(prefix='rwkv-pool-') as folder:
            log_path = Path(folder) / 'pool.log'
            with log_path.open('w') as writer, log_path.open('r') as reader:
                pool = subprocess.Popen(pool_command, cwd=ROOT, stdout=writer,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    wait_for_pool(pool, reader, startup_timeout)
                except Exception:
                    print(log_path.read_text(), file=sys.stderr, flush=True)
                    raise
                train = subprocess.Popen(train_command, cwd=ROOT, start_new_session=True)
                while train.poll() is None:
                    if pool.poll() is not None:
                        print(log_path.read_text(), file=sys.stderr, flush=True)
                        raise RuntimeError(f'teacher pool exited during training (code {pool.returncode})')
                    time.sleep(0.1)
                return train.returncode
    except Interrupted as error:
        return 128 + error.signum
    finally:
        # A second terminal signal must not interrupt cleanup.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            stop_process_group(train, shutdown_timeout)
            stop_process_group(pool, shutdown_timeout)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rwkv-config', required=True)
    parser.add_argument('--startup-timeout', type=float, default=120)
    parser.add_argument('--shutdown-timeout', type=float, default=15)
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Optional training command after --; config and pool flags are appended')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        return launch(args.rwkv_config, command, args.startup_timeout, args.shutdown_timeout)
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        print(f'distillation launcher: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
