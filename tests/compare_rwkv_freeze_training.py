"""Compare frozen production training across DP/TP/PP/SP/CP on fixed samples."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

import torch

from compare_dp_distill import canonical, metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--unfrozen', action='store_true')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--initialization', type=Path)
    parser.add_argument('--cases', nargs='+', default=['combined', 'distopt', 'cp-distopt'])
    args = parser.parse_args()
    suffix = ('trained-' if args.initialization else '') + 'precision-' + ('unfrozen' if args.unfrozen else 'frozen') + ('-fp32' if args.fp32 else '-bf16')
    if args.prepare:
        root = Path(__file__).resolve().parents[1]
        fixture = (args.initialization.resolve() if args.initialization else
                   root / 'training-smoke/audit/cropped-native.pth')
        for case in ['dense'] + args.cases:
            nproc = 1 if case == 'dense' else 8
            folder = args.output / case / suffix
            folder.mkdir(exist_ok=True)
            cfg = yaml.safe_load((args.output / case / 'freeze-reset/config.yaml').read_text())
            cfg.update(load=None, ckpt_step=None, load_pretrained_rwkv=str(fixture),
                       no_load_optim=False, train_iters=1, save=None)
            cfg['model_parallel']['gradient_accumulation_fusion'] = False
            cfg['rwkv_key_fusion'] = False
            if args.unfrozen:
                from rwkv_megatron.model_freeze import FREEZE_COMPONENTS
                cfg.update({f'rwkv_freeze_{name}': False for name in FREEZE_COMPONENTS})
                cfg.update(rwkv_freeze_extra_patterns=[], rwkv_train_extra_patterns=[])
            if args.fp32:
                cfg['model_parallel']['bf16'] = False
                cfg['use_distributed_optimizer'] = False
            config = folder / 'config.yaml'
            config.write_text(yaml.safe_dump(cfg))
            env = dict(os.environ, RWKV_DP_AUDIT_DIR=str(folder.resolve()),
                       RWKV_DP_OPTIMIZER_AUDIT='1', RWKV_TEACHER_SAFETY='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', RWKV_DP_FP32='1' if args.fp32 else '0')
            with (folder / 'training.log').open('w') as log:
                result = subprocess.run([sys.executable, '-m', 'torch.distributed.run',
                    '--standalone', f'--nproc-per-node={nproc}',
                    'tests/run_dp_distill_training.py', '--rwkv-config', str(config.resolve())],
                    env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=240)
            assert result.returncode == 0, (case, folder / 'training.log')
            print('PASS SHARED INITIALIZATION TRAINING', case, flush=True)
    def records(case):
        return [torch.load(p, weights_only=False) for p in
                sorted((args.output / case / suffix).glob('step1-rank*.pt'))]
    dense = records('dense')
    ref_grads = canonical(dense, 'grads', 1, 1)
    ref_weights = canonical(dense, 'before', 1, 1)
    ref_logits = dense[0]['logits']
    ref_loss = dense[0]['losses']
    report = {}
    for case in args.cases:
        rows = records(case)
        assert rows
        # CP ranks hold replicas of the globally reduced parameter gradients.
        cp0 = [r for r in rows if r['cp'] == 0]
        grads = canonical(cp0, 'grads', 2, 2)
        torch.testing.assert_close(canonical(cp0, 'before', 2, 2), ref_weights, rtol=0, atol=0)
        logits = {}
        for r in cp0:
            if r['pp'] == 1 and r['tp'] == 0:
                assert not logits.keys() & r['logits'].keys(), 'DP samples overlap'
                logits.update(r['logits'])
        assert logits.keys() == ref_logits.keys(), 'Global samples differ'
        keys = sorted(logits)
        logit_metric = metric(torch.stack([logits[k] for k in keys]),
                              torch.stack([ref_logits[k] for k in keys]))
        assert logit_metric['relative_l2'] < (1e-5 if args.fp32 else 0.02)
        grad_metrics = {k: metric(g, ref_grads[k]) for k, g in grads.items()}
        significant = {k: m for k, m in grad_metrics.items() if m['reference_l2'] >= 1e-3}
        worst = max(significant, key=lambda k: significant[k]['relative_l2'])
        assert all(m['finite'] for m in grad_metrics.values())
        assert significant[worst]['relative_l2'] < (1e-4 if args.fp32 else 0.05), (case, worst, significant[worst])
        loss = next(r['losses'] for r in cp0 if r['pp'] == 1 and r['tp'] == 0 and r['dp'] == 0)
        loss_errors = {k: abs(loss[k] - ref_loss[k]) / abs(ref_loss[k])
                       for k in ('lm loss', 'distill/kl_loss', 'distill/student_ce_loss')}
        assert max(loss_errors.values()) < (1e-5 if args.fp32 else 0.005)
        # All DP/CP replicas must preserve identical weights after the update.
        for r in rows:
            reference = next(x for x in rows if x['tp'] == r['tp'] and x['pp'] == r['pp'])
            for key in ('before', 'after', 'grads'):
                torch.testing.assert_close(r[key], reference[key], rtol=0, atol=0)
        report[case] = dict(logits_relative_l2=logit_metric['relative_l2'],
                            worst_gradient=worst,
                            worst_gradient_relative_l2=significant[worst]['relative_l2'],
                            loss_relative_errors=loss_errors, replicas_bit_identical=True)
        print('PASS FROZEN TRAINING ALIGNMENT', case, report[case], flush=True)
    (args.output / (suffix + '.json')).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
