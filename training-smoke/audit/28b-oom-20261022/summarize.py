"""Summarize actual training logs and storage audits without loading checkpoints."""
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
result = {}
for phase in ['cp2-no-offload-2k', 'cp2-no-offload-4k-recompute2', 'cp2-no-offload-remainder', 'cp2-no-offload-low-nccl', 'baseline', 'tp4pp2cp1-no-offload', 'reproduce', 'cp2-offload-before-init-fix', 'cp2-offload', 'cp1-control', 'cp2-checkpoint', 'cp2-resume']:
    log = HERE / f'{phase}.log'
    if not log.exists():
        continue
    text = log.read_text(errors='replace')
    steps = []
    for line in text.splitlines():
        match = re.search(r'iteration\s+(\d+)/\s*(\d+)\s*\|.*elapsed time per iteration \(ms\):\s*([\d.]+).*lm loss:\s*([\d.Ee+-]+).*grad norm:\s*([\d.Ee+-]+).*skipped iterations:\s*(\d+).*nan iterations:\s*(\d+)', line)
        if match:
            step, total, ms, loss, grad, skip, nan = match.groups()
            steps.append(dict(step=int(step), total=int(total), ms=float(ms), loss=float(loss),
                              grad_norm=float(grad), skipped=int(skip), nan=int(nan)))
    rows = []
    for path in sorted((HERE / phase).glob('rank*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    evaluations = [dict(step=int(step), loss=float(loss), perplexity=float(ppl))
                   for step, loss, ppl in re.findall(r'validation loss at iteration (\d+)[^\n]*lm loss value:\s*([\d.Ee+-]+)\s*\|\s*lm loss PPL:\s*([\d.Ee+-]+)', text)]
    steady = [s['ms'] for s in steps if s['step'] > 2]
    result[phase] = dict(steps=steps, evaluations=evaluations,
        logged_loss_and_grad_finite=all(math.isfinite(s[k]) for s in steps for k in ['loss', 'grad_norm']) if steps else None,
        median_step_ms_after_step2=statistics.median(steady) if steady else None,
        max_audited_cuda_allocated_gib=max((r['max_allocated'] / 2**30 for r in rows), default=None),
        optimizer_built_rank0=next((r for r in rows if r['rank'] == 0 and r['event'] == 'optimizer_built'), None),
        after_step_rank0=[r for r in rows if r['rank'] == 0 and r['event'] == 'after_step'],
        audited_oom_ranks=sorted(r['rank'] for r in rows if r['event'] == 'oom'),
        saved_iterations=[int(s) for s in re.findall(r'successfully saved checkpoint[^\n]*iteration\s+(\d+)', text)],
        reloaded_iterations=[int(s) for s in re.findall(r'successfully loaded checkpoint[^\n]*iteration\s+(\d+)', text)])
if all(k in result and result[k]['steps'] for k in ['cp1-control', 'cp2-offload']):
    cp1 = {s['step']: s for s in result['cp1-control']['steps']}
    cp2 = {s['step']: s for s in result['cp2-offload']['steps']}
    pairs = [dict(step=i, cp1_loss=cp1[i]['loss'], cp2_loss=cp2[i]['loss'],
                  loss_abs_diff=abs(cp1[i]['loss'] - cp2[i]['loss']),
                  grad_norm_relative_diff=abs(cp1[i]['grad_norm'] - cp2[i]['grad_norm']) / cp1[i]['grad_norm'])
             for i in sorted(cp1.keys() & cp2.keys())]
    result['cp_comparison'] = dict(pairs=pairs,
        max_logged_loss_abs_diff=max(p['loss_abs_diff'] for p in pairs),
        max_logged_grad_norm_relative_diff=max(p['grad_norm_relative_diff'] for p in pairs))
if all(k in result and result[k]['steps'] for k in ['cp2-offload', 'cp2-checkpoint', 'cp2-resume']):
    reference = {s['step']: s for s in result['cp2-offload']['steps']}
    observed = result['cp2-checkpoint']['steps'] + result['cp2-resume']['steps']
    comparisons = [dict(step=s['step'], loss_abs_diff=abs(s['loss'] - reference[s['step']]['loss']),
                        grad_norm_abs_diff=abs(s['grad_norm'] - reference[s['step']]['grad_norm']))
                   for s in observed]
    result['checkpoint_continuation'] = dict(
        comparisons=comparisons,
        losses_equal_at_log_precision=all(c['loss_abs_diff'] == 0 for c in comparisons),
        grad_norms_equal_at_log_precision=all(c['grad_norm_abs_diff'] == 0 for c in comparisons),
        full_model_optimizer_rng_tensor_comparison_performed=False)
(HERE / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
for phase, item in result.items():
    print(phase, {k: v for k, v in item.items() if k not in ['steps', 'after_step_rank0', 'optimizer_built_rank0', 'pairs']})
