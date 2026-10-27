"""Emergency checkpoints resume exactly after a teacher failure before a step."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from compare_dp_distill_resume import exact

ROOT = Path(__file__).resolve().parents[1]
CASES = {'dense': (1, 1, 1, 1, False), 'combined': (2, 2, 2, 1, True),
         'cp-combined': (1, 2, 2, 2, True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'training-smoke/dp2-distill-20261023/dense-bf16.yaml')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    options = parser.parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in options.cases:
        dp, tp, pp, cp, sp = CASES[name]
        base = yaml.safe_load(options.config.read_text())
        base['model_parallel'].update(tensor_model_parallel_size=tp,
                                      pipeline_model_parallel_size=pp,
                                      context_parallel_size=cp, sequence_parallel=sp)
        base.update(deterministic_mode=True, load=None, train_iters=3,
                    save_interval=250, eval_iters=0, rwkv_teacher_step_timeout=15,
                    rwkv_teacher_async_prefetch=False, wandb_project=None)

        def run(tag, *, safety=True, load=None, failure=False, eval_failure=False):
            folder = output/f'{name}-{tag}'
            folder.mkdir(exist_ok=True)
            cfg = dict(base, save=str(folder/'checkpoints'), load=load,
                       eval_iters=1 if eval_failure else 0, eval_interval=2)
            path = folder/'config.yaml'
            path.write_text(yaml.safe_dump(cfg))
            environment = dict(os.environ, RWKV_DP_AUDIT_DIR=str(folder),
                               RWKV_DP_OPTIMIZER_AUDIT='1', RWKV_TEACHER_SAFETY=str(int(safety)))
            if failure:
                environment.update(RWKV_TEACHER_OUTAGE_STEP='2', RWKV_TEACHER_OUTAGE_DP=str(dp-1))
            if eval_failure:
                environment['RWKV_TEACHER_OUTAGE_EVAL'] = '1'
            with (folder/'training.log').open('w') as log:
                result = subprocess.run(
                    [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                     f'--nproc-per-node={dp*tp*pp*cp}', str(ROOT/'tests/run_dp_distill_training.py'),
                     '--rwkv-config', str(path)], cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, timeout=300)
            assert result.returncode == (1 if failure or eval_failure else 0), (folder, result.returncode)
            print('COMPLETE', folder.name, flush=True)
            return folder

        baseline = run('baseline', safety=False)
        failed = run('failure', failure=True)
        emergency = failed/'checkpoints/teacher-emergency'
        metadata = json.loads((emergency/'teacher-failure.json').read_text())
        assert metadata['iteration'] == 1 and metadata['consumed_train_samples'] == 16
        assert not list(failed.glob('step2-*.pt'))
        failure_rows = [json.loads(path.read_text()) for path in failed.glob('failure-rank*.json')]
        assert len(failure_rows) == dp*tp*pp*cp
        assert all(row['step'] == 2 and row['forward_calls'] == 0 for row in failure_rows)
        resumed = run('resumed', load=str(emergency))
        counts = dict(tensors=0, elements=0)
        for step in (2, 3):
            for rank in range(dp*tp*pp*cp):
                expected = torch.load(baseline/f'step{step}-rank{rank}.pt', weights_only=False)
                actual = torch.load(resumed/f'step{step}-rank{rank}.pt', weights_only=False)
                for field in ('before', 'after', 'grads', 'optimizer', 'samples', 'logits', 'grad_norm'):
                    exact(expected[field], actual[field], f'{name}/step{step}/rank{rank}/{field}', counts)
        eval_failed = run('eval-failure', eval_failure=True)
        eval_emergency = eval_failed/'checkpoints/teacher-emergency'
        eval_metadata = json.loads((eval_emergency/'teacher-failure.json').read_text())
        assert eval_metadata['iteration'] == 2 and eval_metadata['consumed_train_samples'] == 32
        assert 'validation before forward' in eval_metadata['reason']
        eval_resumed = run('eval-resumed', load=str(eval_emergency))
        for rank in range(dp*tp*pp*cp):
            expected = torch.load(baseline/f'step3-rank{rank}.pt', weights_only=False)
            actual = torch.load(eval_resumed/f'step3-rank{rank}.pt', weights_only=False)
            for field in ('after', 'grads', 'optimizer', 'samples', 'logits'):
                exact(expected[field], actual[field], f'{name}/eval-resume/rank{rank}/{field}', counts)
        report[name] = dict(training_failure_before_any_forward=True, saved_iteration=1,
                            evaluation_failure_saved_iteration=2, strict_resume_bit_identical=True,
                            compared=counts, layout=dict(dp=dp, tp=tp, pp=pp, cp=cp, sp=sp))
        (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print('PASS', name, report[name], flush=True)


if __name__ == '__main__':
    main()
