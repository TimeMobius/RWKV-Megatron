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
            raise RuntimeError(
                f"teacher pool exited during startup (code {process.returncode})"
            )
        line = log.readline()
        if line:
            print(line, end="", flush=True)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "teacher_pool_ready":
                return event
        else:
            time.sleep(0.05)
    raise TimeoutError("teacher pool startup timed out")


def default_training_command(settings):
    """Infer one-node torchrun size from the configured DP/TP/PP layout."""
    dp, tp, pp = validate_parallel_settings(settings)
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={dp * tp * pp}",
        str(ROOT / "pretrain_rwkv7.py"),
    ]


def validate_parallel_settings(settings):
    """Check the uniform ordinary-PP layouts supported by online distillation."""
    parallel = settings.get("model_parallel", {})
    if not isinstance(parallel, dict):
        raise ValueError("model_parallel must be a mapping")
    dp = parallel.get("data_parallel_size", 1)
    tp = parallel.get("tensor_model_parallel_size", 1)
    pp = parallel.get("pipeline_model_parallel_size", 1)
    cp = parallel.get("context_parallel_size", 1)
    if any(type(size) is not int or size < 1 for size in (dp, tp, pp, cp)):
        raise ValueError("parallel sizes must be positive integers")
    if cp != 1:
        raise ValueError("online distillation does not support context parallelism")
    if (
        parallel.get("virtual_pipeline_model_parallel_size") is not None
        or settings.get("virtual_pipeline_model_parallel_size") is not None
    ):
        raise ValueError("online distillation supports non-interleaved PP only")
    sp = parallel.get("sequence_parallel", False)
    if type(sp) is not bool:
        raise ValueError("sequence_parallel must be a boolean")
    if sp and tp == 1:
        raise ValueError("sequence_parallel requires tensor_model_parallel_size > 1")
    language = settings.get("language_model", {})
    if not isinstance(language, dict):
        raise ValueError("language_model must be a mapping")
    for key, divisor in (("hidden_size", tp), ("num_attention_heads", tp)):
        value = language.get(key)
        if value is not None and (
            type(value) is not int or value < 1 or value % divisor
        ):
            raise ValueError(f"{key} must be a positive integer divisible by TP={tp}")
    layers = language.get("num_layers")
    custom_split = any(
        language.get(key) is not None
        for key in (
            "num_layers_in_first_pipeline_stage",
            "num_layers_in_last_pipeline_stage",
        )
    )
    if layers is not None and (
        type(layers) is not int or layers < pp or (not custom_split and layers % pp)
    ):
        raise ValueError(
            f"num_layers must cover PP={pp} and divide evenly unless a custom split is specified"
        )
    sequence = settings.get("seq_length")
    if sequence is not None and (
        type(sequence) is not int or sequence < 1 or (sp and sequence % tp)
    ):
        raise ValueError(
            "seq_length must be positive and divisible by TP when SP is enabled"
        )
    micro, batch = settings.get("micro_batch_size"), settings.get("global_batch_size")
    for key, value in (("micro_batch_size", micro), ("global_batch_size", batch)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be a positive integer")
    if micro is not None and batch is not None and batch % (micro * dp):
        raise ValueError(
            f"global_batch_size must be divisible by micro_batch_size * DP={micro * dp}"
        )
    return dp, tp, pp


def validate_training_command(settings, command):
    """Check explicit local torchrun sizes while preserving custom executables."""
    dp, tp, pp = validate_parallel_settings(settings)
    is_torchrun = any(
        Path(part).name in ("torchrun", "torch.distributed.run") for part in command
    )
    if not is_torchrun:
        return
    count = None
    for index, part in enumerate(command):
        if part.startswith(("--nproc-per-node=", "--nproc_per_node=")):
            count = part.split("=", 1)[1]
        elif part in ("--nproc-per-node", "--nproc_per_node") and index + 1 < len(
            command
        ):
            count = command[index + 1]
    if count is None or not count.isdecimal() or int(count) < 1:
        raise ValueError(
            "local torchrun requires an explicit positive integer --nproc-per-node"
        )
    requested = int(count)
    model_size = tp * pp
    if requested % model_size:
        raise ValueError("torchrun process count must be divisible by TP * PP")
    if (
        "data_parallel_size" in settings.get("model_parallel", {})
        and requested != dp * model_size
    ):
        raise ValueError(
            "torchrun process count disagrees with configured DP * TP * PP"
        )
    actual_dp = requested // model_size
    actual = {
        **settings,
        "model_parallel": {
            **settings.get("model_parallel", {}),
            "data_parallel_size": actual_dp,
        },
    }
    validate_parallel_settings(actual)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [
            value.strip()
            for value in visible.split(",")
            if value.strip() and value.strip() != "-1"
        ]
        if len(set(devices)) != len(devices):
            raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate devices")
        if requested > len(devices):
            raise ValueError(
                f"torchrun requires {requested} GPUs but CUDA_VISIBLE_DEVICES exposes {len(devices)}"
            )
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        available = len(result.stdout.strip().splitlines())
        if requested > available:
            raise ValueError(
                f"torchrun requires {requested} GPUs but this node has {available}"
            )


def launch(
    config, command=None, startup_timeout=120, shutdown_timeout=15, *, pool_command=None
):
    if startup_timeout <= 0 or shutdown_timeout <= 0:
        raise ValueError("startup and shutdown timeouts must be positive")
    config = Path(config).resolve()
    settings = yaml.safe_load(config.read_text())
    if not settings.get("rwkv_distill", False):
        raise ValueError("launcher requires rwkv_distill: true")
    command = command or default_training_command(settings)
    validate_training_command(settings, command)
    host = settings.get("rwkv_teacher_pool_host", "127.0.0.1")
    port = settings.get("rwkv_teacher_pool_port", 50055)
    # Refuse an existing listener so a failed new pool never attaches training
    # to an unrelated server. Only processes started here are terminated.
    try:
        with socket.create_connection((host, port), timeout=0.5):
            raise RuntimeError(f"teacher pool address is already in use: {host}:{port}")
    except (ConnectionRefusedError, TimeoutError):
        pass
    pool_command = pool_command or [
        sys.executable,
        "-u",
        str(ROOT / "tools/teacher_pool_server.py"),
        "--rwkv-config",
        str(config),
    ]
    train_command = [*command, "--rwkv-config", str(config), "--rwkv-teacher-pool"]
    pool = train = None
    old_handlers = {}

    def interrupt(signum, frame):
        raise Interrupted(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupt)
    try:
        # Separate reader/writer handles avoid shared seek offsets between the
        # subprocess and the parent. Keep the log to report startup failures.
        with tempfile.TemporaryDirectory(prefix="rwkv-pool-") as folder:
            log_path = Path(folder) / "pool.log"
            with log_path.open("w") as writer, log_path.open("r") as reader:
                pool = subprocess.Popen(
                    pool_command,
                    cwd=ROOT,
                    stdout=writer,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    wait_for_pool(pool, reader, startup_timeout)
                except Exception:
                    print(log_path.read_text(), file=sys.stderr, flush=True)
                    raise
                train = subprocess.Popen(
                    train_command, cwd=ROOT, start_new_session=True
                )
                while train.poll() is None:
                    if pool.poll() is not None:
                        print(log_path.read_text(), file=sys.stderr, flush=True)
                        raise RuntimeError(
                            f"teacher pool exited during training (code {pool.returncode})"
                        )
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
    parser.add_argument("--rwkv-config", required=True)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--shutdown-timeout", type=float, default=15)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional training command after --; config and pool flags are appended",
    )
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        return launch(
            args.rwkv_config, command, args.startup_timeout, args.shutdown_timeout
        )
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        print(f"distillation launcher: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
