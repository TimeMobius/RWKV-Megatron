"""End-to-end freeze strategy changes on the real sampler/DDP/Adam/checkpoint.

Run via the normal CUDA environment wrapper. All generated artifacts are local
under --output. No teacher HTTP or W&B. Tests ordinary and distributed Adam,
DP/TP/PP/SP/CP, rejected unsafe loads, reset and subsequent exact resume.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from compare_dp_distill_resume import exact

ROOT = Path(__file__).resolve().parents[1]
POLICY = dict(rwkv_freeze_embedding=True, rwkv_freeze_lm_head=True,
              rwkv_freeze_attn_norm=True, rwkv_freeze_ffn_norm=True,
              rwkv_freeze_final_norm=True, rwkv_freeze_ffn_value=True,
              rwkv_freeze_attn_key=True, rwkv_freeze_attn_value=True,
              rwkv_freeze_attn_output=True, rwkv_freeze_token_shift=True,
              rwkv_freeze_lora=True, rwkv_freeze_extra_patterns=['blocks.2.*'],
              rwkv_train_extra_patterns=['blocks.2.att.key.weight'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--cases', nargs='+', default=['dense', 'combined', 'distopt', 'cp-distopt', 'frozen-stage'])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / 'configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-single.yaml').read_text())
    base['language_model'].update(num_layers=4, hidden_size=256, num_attention_heads=4,
        ffn_hidden_size=1024, D_AAA_LORA=16, D_DECAY_LORA=16, D_GATE_LORA=16, D_MV_LORA=16)
    base.update(tokenizer_type='NullTokenizer', tokenizer_model=None, vocab_size=119,
        make_vocab_size_divisible_by=16, data_path=None, mock_data=True, split='80,10,10',
        data_cache_path=str(output / 'cache'), num_workers=0,
        load_pretrained_rwkv=str(ROOT / 'training-smoke/audit/cropped-native.pth'),
        lr=1e-4, min_lr=1e-4, rwkv_distill_valid_vocab_size=119, rwkv_distill_ce_weight=0.3,
        rwkv_teacher_async_prefetch=False, rwkv_key_fusion=False, deterministic_mode=True, log_throughput=False,
        wandb_project=None, wandb_exp_name=None, tensorboard_dir=None,
        wandb_save_dir=str(output / 'wandb'))
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for case in args.cases:
        tp, pp, cp, nproc, fmt, distopt = {
            'dense': (1, 1, 1, 1, 'torch', False),
            'combined': (2, 2, 1, 8, 'torch', False),
            'distopt': (2, 2, 1, 8, 'torch_dist', True),
            'cp-distopt': (2, 2, 2, 8, 'torch_dist', True),
            'frozen-stage': (2, 2, 1, 8, 'torch_dist', True),
        }[case]
        policy = (dict(rwkv_freeze_embedding=True,
                       rwkv_freeze_extra_patterns=['decoder.layers.[01].*', 'pre_norm.*'])
                  if case == 'frozen-stage' else POLICY)
        folder = output / case
        folder.mkdir(exist_ok=True)
        def run(name, policy, load=None, step=None, reset=False, rejected=False):
            target = folder / name
            target.mkdir(exist_ok=True)
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg.update(train_iters=4 if name != 'baseline' else 2, lr_decay_iters=8,
                       eval_interval=100, eval_iters=0, global_batch_size=8,
                       eval_global_batch_size=8, micro_batch_size=1,
                       eval_micro_batch_size=1, save_interval=1,
                       save=str(target / 'checkpoints'), load=load,
                       ckpt_step=step, no_load_optim=reset, finetune=False,
                       ckpt_format=fmt, use_distributed_optimizer=distopt,
                       log_progress=False, log_memory_to_tensorboard=False,
                       rwkv_teacher_step_timeout=15, distributed_timeout_minutes=3,
                       **policy)
            cfg['model_parallel'].update(tensor_model_parallel_size=tp,
                                         pipeline_model_parallel_size=pp,
                                         context_parallel_size=cp,
                                         sequence_parallel=(tp > 1), gradient_accumulation_fusion=False)
            cfg['language_model'].update(recompute_granularity='full',
                                         recompute_method='uniform', recompute_num_layers=1)
            path = target / 'config.yaml'
            env = dict(os.environ, RWKV_DP_AUDIT_DIR=str(target),
                       RWKV_DP_OPTIMIZER_AUDIT='1', RWKV_TEACHER_SAFETY='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
            if args.verify_only:
                text = (target / 'training.log').read_text()
                if rejected:
                    assert 'RWKV freeze strategy differs' in text
                else:
                    assert '[after training is done]' in text
                return target
            path.write_text(yaml.safe_dump(cfg))
            with (target / 'training.log').open('w') as log:
                result = subprocess.run([sys.executable, '-m', 'torch.distributed.run',
                    '--standalone', f'--nproc-per-node={nproc}',
                    'tests/run_dp_distill_training.py', '--rwkv-config', str(path)],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
            text = (target / 'training.log').read_text()
            if rejected:
                assert result.returncode != 0 and 'RWKV freeze strategy differs' in text, (case, name, target)
                assert not list(target.glob('step*-rank*.pt')), 'Unsafe load reached a training step'
            else:
                assert result.returncode == 0, (case, name, target / 'training.log')
            print('PASS TRAINING', case, name, flush=True)
            return target

        baseline = run('baseline', {})
        source = str(baseline / 'checkpoints')
        run('reject-freeze', policy, source, 2, rejected=True)
        frozen = run('freeze-reset', policy, source, 2, reset=True)
        frozen_source = str(frozen / 'checkpoints')
        resumed = run('same-resume', policy, frozen_source, 3)
        run('reject-unfreeze', {}, frozen_source, 3, rejected=True)
        unfrozen = run('unfreeze-reset', {}, frozen_source, 3, reset=True)
        counts = dict(tensors=0, elements=0)
        for rank in range(nproc):
            changed = torch.load(frozen / f'step3-rank{rank}.pt', weights_only=False)
            reverse = torch.load(unfrozen / f'step4-rank{rank}.pt', weights_only=False)
            original = torch.load(baseline / f'step2-rank{rank}.pt', weights_only=False)
            exact(changed['before'], original['after'], f'{case}/rank{rank}/loaded-model-freeze', counts)
            exact(reverse['before'], changed['after'], f'{case}/rank{rank}/loaded-model-unfreeze', counts)
            assert changed['skipped'] == reverse['skipped'] == 0
            frozen_keys = changed['before'].keys() - set(changed['trainable'])
            if case != 'frozen-stage':
                assert frozen_keys
            assert not (reverse['before'].keys() - set(reverse['trainable']))
            for key in frozen_keys:
                assert torch.equal(changed['before'][key], changed['after'][key]), key
                assert changed['grads'][key].count_nonzero() == 0, key
            def adam_steps(item):
                children = item['optimizer']
                if isinstance(children, dict):
                    children = [children]
                steps = []
                for child in children:
                    for state in child['state']:
                        steps.extend(group['step'] for group in state['param_groups'] if 'step' in group)
                return steps
            if changed['trainable']:
                assert adam_steps(changed) and set(adam_steps(changed)) == {1}
            else:
                children = changed['optimizer']
                children = children if isinstance(children, list) else [children]
                # A stub can carry empty Adam groups and increment a group step;
                # it must never allocate slots or FP32 master weights.
                assert all(not state['state'] and all(not group['params']
                           for group in state['param_groups'])
                           for child in children for state in child['state'])
                assert all(not group for child in children for group in child['masters'])
            assert adam_steps(reverse) and set(adam_steps(reverse)) == {1}
            before = torch.load(frozen / f'step4-rank{rank}.pt', weights_only=False)
            after = torch.load(resumed / f'step4-rank{rank}.pt', weights_only=False)
            for key in ('samples', 'logits', 'grads', 'before', 'after', 'optimizer',
                        'skipped', 'grad_norm', 'trainable'):
                exact(before[key], after[key], f'{case}/rank{rank}/{key}', counts)
            if rank == 0:
                assert not set(changed['samples']) & set(before['samples']), 'Data position repeated'
        for label, run_dir, step in [('freeze-reset', frozen, 3), ('same-resume', resumed, 4),
                                     ('unfreeze-reset', unfrozen, 4)]:
            # Use Megatron's common-state reader for both distributed checkpoint formats.
            root = run_dir / f'checkpoints/iter_{step:07d}'
            if fmt == 'torch':
                paths = list(root.glob('*/model_optim_rng.pt'))
                assert paths
                states = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
            else:
                from megatron.core.dist_checkpointing import load_common_state_dict
                states = [load_common_state_dict(root)]
            for state in states:
                assert state['iteration'] == step
                assert state['args'].consumed_train_samples == step * 8
                assert state['opt_param_scheduler']['num_steps'] == (step - 2 if label != 'unfreeze-reset' else 1) * 8
        report[case] = dict(passed=True, exact_resume=True, frozen_weights_unchanged=True,
                            strategy_change_rejected=True, adam_reset_both_directions=True,
                            data_position_preserved=True, **counts)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('PASS CASE', case, report[case], flush=True)


if __name__ == '__main__':
    main()
