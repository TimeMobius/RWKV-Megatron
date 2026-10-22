"""Import recorded training metrics; original timestamps remain explicit fields."""

import bisect
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import wandb
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
RUN_ID = 'cp4-halfday-20261021'


def main():
    state = json.loads((HERE / 'status.json').read_text())
    assert state['status'] == 'completed'
    config = yaml.safe_load((HERE / 'config-train.yaml').read_text())
    history = defaultdict(dict)
    tags = {
        'lm loss': 'train/loss_instant',
        'grad-norm': 'train/grad_norm',
        'learning-rate': 'train/learning_rate',
        'iteration-time': 'performance/step_seconds_interval_mean',
        'throughput': 'performance/tflops_per_second_per_gpu',
        'mem-reserved-bytes': 'memory/rank0_reserved_bytes',
        'mem-allocated-bytes': 'memory/rank0_allocated_bytes',
        'mem-max-allocated-bytes': 'memory/rank0_max_allocated_bytes',
        'lm loss validation': 'validation/loss',
        'lm loss validation ppl': 'validation/perplexity',
    }
    for event_file in (HERE / 'tensorboard/train').rglob('events.out.tfevents.*'):
        events = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
        events.Reload()
        for tag, metric in tags.items():
            if tag not in events.Tags()['scalars']:
                continue
            for event in events.Scalars(tag):
                assert math.isfinite(event.value), (tag, event.step)
                history[event.step][metric] = event.value
                history[event.step]['original_timestamp_unix'] = event.wall_time

    times = []
    steps = []
    log_times = []
    for line in (HERE / 'train.log').read_text().splitlines():
        match = re.search(r'\[([^]]+)\] iteration\s+(\d+)/', line)
        if not match or 'elapsed time per iteration' not in line:
            continue
        step = int(match[2])
        timestamp = datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S.%f').timestamp()
        steps.append(step)
        log_times.append(timestamp)
        for label, metric in [
            ('lm loss', 'train/loss_interval_mean'),
            ('number of skipped iterations', 'train/skipped_in_log_interval'),
            ('number of nan iterations', 'train/nan_in_log_interval'),
        ]:
            value = re.search(rf'{label}:\s*([\d.Ee+-]+)', line)
            history[step][metric] = float(value[1])
        value = re.search(r'elapsed time per iteration \(ms\):\s*([\d.]+)', line)
        ms = float(value[1])
        history[step]['performance/tokens_per_second_global'] = 32768000 / ms
        if step >= 100:
            times.append(ms)

    for key in ['baseline_validation', 'checkpoint_validation']:
        for row in state[key]:
            history[row['iteration']]['validation/loss'] = row['loss']
            history[row['iteration']]['validation/perplexity'] = math.exp(row['loss'])
    # Aggregate GPU samples onto the preceding logged step, omitting eval-only stages.
    gpu = defaultdict(list)
    for row in csv.reader((HERE / 'gpu.csv').open()):
        timestamp = datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
        if timestamp < log_times[0] or timestamp > log_times[-1]:
            continue
        step = steps[bisect.bisect_right(log_times, timestamp) - 1]
        gpu[step].append((int(row[1]), *map(float, row[2:])))
    peak_memory = 0
    for step, samples in gpu.items():
        for rank in range(8):
            rank_samples = [r for r in samples if r[0] == rank]
            if not rank_samples:
                continue
            peak_memory = max(peak_memory, max(r[1] for r in rank_samples))
            for column, metric in [(1, 'memory_mib'), (2, 'utilization_percent'), (3, 'power_watts')]:
                history[step][f'gpu_{rank}/{metric}'] = statistics.mean(r[column] for r in rank_samples)

    provenance = json.loads((HERE / 'provenance.json').read_text())
    rows = [{'iteration': step, 'consumed_tokens': step * 32768, **values}
            for step, values in sorted(history.items())]
    (HERE / 'wandb-import-history.jsonl').write_text(
        ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
    run = wandb.init(
        entity='mobius', project='Mobius-r7-28b-qwen-pretrain', id=RUN_ID,
        name='rwkv7-2p7b-tp2-cp4-sp-16k-halfday-20261021',
        resume='allow', mode='online', dir=str(HERE),
        config={'training': config, 'provenance': provenance, 'historical_import': True,
                'original_started_at_utc': state['train_started_at_utc'],
                'original_finished_at_utc': state['finished_at_utc'],
                'loss_instant': 'TensorBoard loss at each 10th iteration',
                'loss_interval_mean': 'stdout mean over each 10-iteration interval',
                'gpu_alignment': 'recorded samples mapped to preceding logged train iteration'},
        notes='Historical import after the 12-hour job completed. Original runtime is in config; '
              'W&B upload runtime is not training runtime. No checkpoint tensors uploaded.',
        tags=['cp4', 'sp', '16k', 'longrun', 'historical-import'],
        settings=wandb.Settings(disable_git=True, x_disable_stats=True, init_timeout=60),
    )
    run.define_metric('iteration')
    run.define_metric('*', step_metric='iteration')
    for row in rows:
        run.log(row, step=row['iteration'])
    summary = {
        'completed_train_iterations': state['final_checkpoint_iteration'],
        'consumed_tokens_total': state['final_checkpoint_iteration'] * 32768,
        'baseline_validation_loss': state['baseline_validation'][0]['loss'],
        'final_checkpoint_validation_loss': state['checkpoint_validation'][0]['loss'],
        'total_skipped_iterations_in_logs': state['skipped_iterations_in_logged_intervals'],
        'total_nan_iterations_in_logs': state['nan_iterations_in_logged_intervals'],
        'logged_loss_and_grad_norm_finite': state['logged_losses_and_grad_norms_finite'],
        'steady_median_step_ms_from_iteration_100': statistics.median(times),
        'sampled_peak_gpu_memory_mib': peak_memory,
        'checkpoint_iterations': state['completed_checkpoint_iterations'],
        'historical_import': True,
        'production_sources_unchanged': state['recorded_production_sources_unchanged'],
    }
    run.summary.update(summary)
    url = run.url
    run.finish()
    (HERE / 'wandb-upload.json').write_text(json.dumps({
        'entity': 'mobius', 'project': 'Mobius-r7-28b-qwen-pretrain', 'run_id': RUN_ID,
        'url': url, 'history_rows': len(rows), 'summary': summary,
    }, indent=2) + '\n')
    print(url)


if __name__ == '__main__':
    main()
