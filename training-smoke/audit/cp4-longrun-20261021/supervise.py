"""Run baseline evaluation, a bounded training job, and checkpoint evaluation."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from summarize import atomic_write, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    files = [ROOT / 'pretrain_rwkv7.py', ROOT / 'rwkv_megatron/rwkv7_layer_specs.py',
             ROOT / 'rwkv_megatron/token_shift.py']
    fla = ROOT / '3rdparty/flash-linear-attention-260115'
    files += [fla / name for name in [
        'fla/ops/cp/context.py', 'fla/ops/generalized_delta_rule/dplr/chunk.py',
        'fla/ops/generalized_delta_rule/dplr/context_parallel.py',
        'fla/ops/generalized_delta_rule/dplr/backends/tilelang/__init__.py']]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def update(state):
    atomic_write(HERE / 'state.json', json.dumps(state, indent=2) + '\n')
    summarize(state)


def phase(name, config, state):
    state.update(phase=name)
    with (HERE / f'{name}.log').open('w') as output:
        command = [str(Path(sys.executable).with_name('torchrun')), '--standalone',
                   '--nproc-per-node=8', str(HERE / 'entry.py'), '--rwkv-config', str(config)]
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        state.update(child_pid=child.pid)
        update(state)
        while True:
            try:
                gpu = subprocess.run([
                    'nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
                with (HERE / 'gpu.csv').open('a') as metrics:
                    metrics.write(gpu.stdout)
            except (subprocess.SubprocessError, OSError) as error:
                print(f'GPU sampling failed: {error}', flush=True)
            update(state)
            try:
                returncode = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
        state.setdefault('exit_codes', {})[name] = returncode
        update(state)
        if returncode:
            raise RuntimeError(f'{name} exited with code {returncode}; see {name}.log')


def main():
    lock = (HERE / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (HERE / 'state.json').exists():
        raise RuntimeError('Existing run state found; create a new output directory for a new run.')
    if any((HERE / 'checkpoints').glob('*')):
        raise RuntimeError('Checkpoint output is not empty.')
    initial_sources = sources()
    state = {'status': 'running', 'phase': 'initializing', 'started_at_utc': now(),
             'supervisor_pid': os.getpid(), 'sources_sha256_start': initial_sources}
    update(state)
    try:
        phase('baseline', HERE / 'config-baseline.yaml', state)
        state['train_started_at_utc'] = now()
        phase('train', HERE / 'config-train.yaml', state)
        tracker = HERE / 'checkpoints/latest_checkpointed_iteration.txt'
        iteration = int(tracker.read_text().strip())
        state['final_checkpoint_iteration'] = iteration
        config = yaml.safe_load((HERE / 'config-baseline.yaml').read_text())
        config.update(load=str(HERE / 'checkpoints'), load_pretrained_rwkv=None,
                      tensorboard_dir=str(HERE / 'tensorboard/final-eval'))
        final_config = HERE / 'config-final-eval.yaml'
        final_config.write_text(yaml.safe_dump(config, sort_keys=False))
        phase('final-eval', final_config, state)
        state.update(status='completed', phase='finished', finished_at_utc=now())
    except BaseException as error:
        state.update(status='failed', error=str(error), finished_at_utc=now())
        traceback.print_exc()
        raise
    finally:
        state['sources_sha256_end'] = sources()
        state['recorded_production_sources_unchanged'] = initial_sources == sources()
        update(state)


if __name__ == '__main__':
    main()
