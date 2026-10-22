"""Summarize all-GPU trials, keeping failed trials visible."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / 'summary.json').read_text())
result = {}
phases = ['cp2-no-offload-remainder', 'cp2-no-offload-low-nccl',
          'cp2-no-offload-2k', 'cp2-no-offload-4k-recompute2']
for phase in phases:
    if phase not in summary:
        continue
    item = summary[phase]
    steps = item['steps']
    peaks = {str(i): 0 for i in range(8)}
    file = HERE / f'{phase}-gpu.csv'
    if file.exists():
        for row in csv.reader(file.open()):
            try:
                peaks[str(int(row[1]))] = max(peaks[str(int(row[1]))], int(row[2]))
            except (ValueError, IndexError):
                continue
    finished = []
    for path in (HERE / phase).glob('rank*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['event'] == 'after_step' and row['step'] == 30:
                finished.append(row)
    all_cuda = all('/cuda/' in key for row in finished for key in row['storage_bytes']) if finished else None
    result[phase] = dict(
        completed_train_steps=len(steps),
        completed_rank_count_at_step30=len(finished),
        passed_30_steps_and_evaluation=(len(steps) == 30 and len(finished) == 8
                                        and bool(item['evaluations']) and not item['audited_oom_ranks']),
        logged_loss_and_grad_finite=item['logged_loss_and_grad_finite'],
        logged_nan_count=sum(row['nan'] for row in steps),
        logged_skipped_count=sum(row['skipped'] for row in steps),
        all_recorded_storage_on_cuda=all_cuda,
        storage_rank0_at_step30=next((row['storage_bytes'] for row in finished if row['rank'] == 0),None),
        median_step_ms_after_step2=item['median_step_ms_after_step2'],
        max_audited_cuda_allocated_gib=item['max_audited_cuda_allocated_gib'],
        sampled_gpu_peak_mib_by_rank=peaks,
        final_training_loss=steps[-1]['loss'] if steps else None,
        evaluations=item['evaluations'],
        audited_oom_ranks=item['audited_oom_ranks'],
    )
if result.get('cp2-no-offload-4k-recompute2',{}).get('passed_30_steps_and_evaluation'):
    a={row['step']:row for row in summary['cp2-offload']['steps']}
    b={row['step']:row for row in summary['cp2-no-offload-4k-recompute2']['steps']}
    result['4k_vs_offload_logged_comparison']=dict(
        max_loss_abs_difference=max(abs(a[i]['loss']-b[i]['loss']) for i in a.keys() & b.keys()),
        max_grad_norm_relative_difference=max(abs(a[i]['grad_norm']-b[i]['grad_norm'])/a[i]['grad_norm']
                                             for i in a.keys() & b.keys()),
        note='Also changes master representation, NCCL configuration and recomputation grouping; not bitwise equivalence.')
(HERE/'cp2-no-offload-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
