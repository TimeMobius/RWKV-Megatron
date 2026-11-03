"""Real RWKV BF16 forward/backward freeze audit, torchrun with eight GPUs.

Reports go to --output, never into tracked config/docs. Uses the existing
cropped model fixture and the production KL/pipeline schedule audit harness.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import run_parallel_distill_model as audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policies', nargs='+')
    parser.add_argument('--layouts', nargs='+')
    parser.add_argument('--fp32', action='store_true')
    opts = parser.parse_args()
    audit.OUT = opts.output
    audit.OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32 = False
    if dist.get_rank() == 0:
        audit.audit.prepare_fixture()
    dist.barrier()
    policies = {
        'embedding-head': dict(rwkv_freeze_embedding=True, rwkv_freeze_lm_head=True),
        'granular': dict(rwkv_freeze_embedding=True, rwkv_freeze_lm_head=True,
                         rwkv_freeze_attn_norm=True, rwkv_freeze_ffn_norm=True,
                         rwkv_freeze_final_norm=True, rwkv_freeze_ffn_value=True,
                         rwkv_freeze_attn_key=True, rwkv_freeze_attn_value=True,
                         rwkv_freeze_attn_output=True, rwkv_freeze_token_shift=True,
                         rwkv_freeze_lora=True,
                         rwkv_freeze_extra_patterns=['blocks.2.*'],
                         rwkv_train_extra_patterns=['blocks.2.att.key.weight']),
    }
    policies.update({
        'inverse': dict(rwkv_freeze_ffn_key=True, rwkv_freeze_attn_receptance=True),
        'frozen-first-stage': dict(rwkv_freeze_embedding=True,
            rwkv_freeze_extra_patterns=['decoder.layers.[01].*', 'pre_norm.*']),
    })
    if opts.policies:
        policies = {name: policies[name] for name in opts.policies}
    layouts = [('dense', 1, 1, False, False), ('tp2', 2, 1, False, False),
               ('pp2', 1, 2, False, False), ('combined', 2, 2, True, False),
               ('recompute', 2, 2, True, True)]
    if opts.layouts:
        layouts = [case for case in layouts if case[0] in opts.layouts]
        assert layouts[0][0] == 'dense', 'Include dense reference first'
    report = {}
    for layout, tp, pp, sp, recompute in layouts:
        audit.run(f'{layout}-unfrozen', tp, pp, sp, recompute=recompute, fp32=opts.fp32)
        for policy, values in policies.items():
            name = f'{layout}-{policy}'
            audit.run(name, tp, pp, sp, recompute=recompute, freeze_args=SimpleNamespace(**values), fp32=opts.fp32)
            if dist.get_rank() == 0:
                actual = torch.load(audit.OUT / f'{name}.pt', weights_only=False)
                reference = torch.load(audit.OUT / f'{layout}-unfrozen.pt', weights_only=False)
                dense = torch.load(audit.OUT / f'dense-{policy}.pt', weights_only=False)
                local_logits = audit.audit.metric(actual['logits'], reference['logits'])
                local_loss = max(abs(float(a['lm loss'][0] / a['lm loss'][1]
                                          - b['lm loss'][0] / b['lm loss'][1]))
                                 for a, b in zip(actual['losses'], reference['losses']))
                # Removing an entire prefix's autograd graph can select a different
                # BF16 execution path. Keep the existing model-audit thresholds;
                # FP32 recurrence separately checks the arithmetic more strictly.
                assert local_logits['relative_l2'] < (1e-5 if opts.fp32 else 0.02)
                assert local_loss < (2e-5 if opts.fp32 else 0.005)
                if policy != 'frozen-first-stage':
                    torch.testing.assert_close(actual['logits'], reference['logits'], rtol=0, atol=0)
                frozen = set(actual['frozen_names'])
                assert frozen == set(dense['frozen_names'])
                local_worst = cross_worst = 0.0
                for key, grad in actual['grads'].items():
                    if key in frozen:
                        assert grad.count_nonzero() == 0, key
                        continue
                    local = audit.audit.metric(grad, reference['grads'][key])
                    cross = audit.audit.metric(grad, dense['grads'][key])
                    assert local['finite'] and cross['finite']
                    if local['reference_l2'] >= 1e-3:
                        local_worst = max(local_worst, local['relative_l2'])
                        assert local['relative_l2'] < (1e-4 if opts.fp32 else 0.05), (name, key, local)
                    if cross['reference_l2'] >= 1e-3:
                        cross_worst = max(cross_worst, cross['relative_l2'])
                        assert cross['relative_l2'] < (1e-4 if opts.fp32 else 0.05), (name, key, cross)
                logits = audit.audit.metric(actual['logits'], dense['logits'])
                assert logits['relative_l2'] < (1e-5 if opts.fp32 else 0.02)
                report[name] = dict(frozen_tensors=sum(k in frozen for k in actual['grads']),
                                    logits_relative_l2=logits['relative_l2'],
                                    same_layout_logits_relative_l2=local_logits['relative_l2'],
                                    same_layout_loss_abs=local_loss,
                                    same_layout_gradient_relative_l2=local_worst,
                                    dense_gradient_relative_l2=cross_worst)
                (audit.OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print('PASS FREEZE ALIGNMENT', name, report[name], flush=True)
            dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
