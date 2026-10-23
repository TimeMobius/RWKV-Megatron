"""Real model/production forward_step numerical audit; launch on eight GPUs.

Artifacts are local .pt tensors plus a compact JSON report. No teacher HTTP.
"""
import argparse
import gc
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training-smoke/audit'))
import distributed as audit
from megatron.core import parallel_state as ps
from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars
from rwkv_conversion import convert_pipeline_stage
from rwkv_megatron.online_teacher import TeacherTopK
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model
import pretrain_rwkv7 as pretrain

OUT = ROOT / 'training-smoke/parallel-distill-20261023'


class NoTimer:
    def start(self, *a, **kw): pass
    def stop(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __enter__(self): return self
    def __exit__(self, *a): pass


class FrozenTeacher:
    prompt_mode = 'token_ids'

    def __init__(self):
        self.calls = 0

    def get_logits(self, tokens, **kwargs):
        self.calls += 1
        # Targets straddle all TP shards, with missing slots and a missing row.
        ids = (tokens[..., None] + torch.tensor([0, 31, 67, 98], device=tokens.device)) % 119
        lp = -torch.tensor([0.2, 1.1, 2.3, 4.0], device=tokens.device).expand_as(ids).clone()
        ids[0, 3] = -1
        return TeacherTopK(ids, lp)


def run(name, tp, pp, sp, recompute=False, fp32=False):
    ps.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    model_parallel_cuda_manual_seed(1234)
    args = SimpleNamespace(transformer_impl='local', rwkv_distill=True, tensor_model_parallel_size=tp,
                           pipeline_model_parallel_size=pp, context_parallel_size=1,
                           virtual_pipeline_model_parallel_size=None, rwkv_teacher_async_prefetch=False,
                           create_attention_mask_in_dataloader=False, micro_batch_size=2, seq_length=128,
                           rwkv_distill_temperature=2.0, rwkv_distill_logit_weight=1.,
                           rwkv_distill_ce_weight=0.3, rwkv_distill_valid_vocab_size=119,
                           check_for_nan_in_loss_and_grad=True)
    global_vars._GLOBAL_ARGS = args
    pretrain.get_timers = lambda: NoTimer()
    pretrain.stimer = NoTimer()
    teacher = FrozenTeacher()
    pretrain._get_online_teacher = lambda args: teacher
    cfg = audit.config_for(tp, pp, sp, recompute, fp32=fp32)
    model = RWKV7Model(cfg, transformer_layer_spec=get_rwkv7_layer_spec(), vocab_size=128,
                       max_sequence_length=128, pre_process=ps.is_pipeline_first_stage(),
                       post_process=ps.is_pipeline_last_stage(), parallel_output=True,
                       share_embeddings_and_output_weights=False, position_embedding_type='none').cuda().to(cfg.params_dtype)
    model.from_rwkv_lm(torch.load(audit.OUT / 'cropped-native.pth', weights_only=True), False)
    if fp32:
        for layer in model.decoder.layers:
            layer.self_attention.core_attention.forward = audit.recurrence
    model.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    model.train()
    generator = torch.Generator().manual_seed(20261023)
    batches = []
    for _ in range(4):
        window = torch.randint(1, 119, (2, 129), generator=generator)
        mask = torch.ones(2, 128)
        mask[1, 100:] = 0
        batches.append(dict(tokens=window[:, :-1].contiguous(), labels=window[:, 1:].contiguous(),
                            loss_mask=mask, position_ids=torch.arange(128).expand(2, -1),
                            attention_mask=None))
    outputs, losses = [], []

    def forward(iterator, module):
        output, loss_fn = pretrain.forward_step(iterator, module)
        if module.post_process:
            shard = output[0].detach().contiguous()
            shards = [torch.empty_like(shard) for _ in range(tp)]
            dist.all_gather(shards, shard, group=ps.get_tensor_model_parallel_group())
            outputs.append(torch.cat(shards, -1).cpu())
            def record(out):
                loss, count, metrics = loss_fn(out)
                losses.append({key: value.detach().cpu() for key, value in metrics.items() if not key.startswith(('teacher/', 'memory/'))})
                return loss, count, metrics
            return output, record
        return output, loss_fn

    get_forward_backward_func()(forward_step_func=forward, data_iterator=iter(batches), model=model,
                                num_microbatches=4, seq_length=128, micro_batch_size=2, forward_only=False)
    _allreduce_non_tensor_model_parallel_grads([model], cfg, ps.get_tensor_model_parallel_group())
    expected_calls = 4 if ps.is_pipeline_last_stage() and ps.get_tensor_model_parallel_rank() == 0 else 0
    assert teacher.calls == expected_calls, (name, dist.get_rank(), teacher.calls, expected_calls)
    grads = {key: p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p).cpu()
             for key, p in model.named_parameters()}
    payload = dict(tp=ps.get_tensor_model_parallel_rank(), pp=ps.get_pipeline_model_parallel_rank(),
                   dp=ps.get_data_parallel_rank(), grads=grads, outputs=outputs, losses=losses)
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(payload, gathered, dst=0)
    if dist.get_rank() == 0:
        selected = [item for item in gathered if item['dp'] == 0]
        merged = {}
        convert_cfg = SimpleNamespace(num_layers=4, hidden_size=256, num_attention_heads=4,
                                      tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
        for stage in range(pp):
            shards = sorted([item for item in selected if item['pp'] == stage], key=lambda item: item['tp'])
            merged.update(convert_pipeline_stage([item['grads'] for item in shards], stage, convert_cfg, 0))
        final = next(item for item in selected if item['pp'] == pp - 1 and item['tp'] == 0)
        torch.save(dict(grads=merged, logits=torch.stack(final['outputs']), losses=final['losses']), OUT / f'{name}.pt')
        print(f'PASS MODEL {name}', flush=True)
    dist.barrier()
    del model, grads, gathered, payload
    gc.collect()
    torch.cuda.empty_cache()
    ps.destroy_model_parallel()


def compare(names):
    results = {}
    for name in names:
        ref = torch.load(OUT / ('dense-fp32.pt' if 'fp32' in name else 'dense.pt'), weights_only=False)
        actual = torch.load(OUT / f'{name}.pt', weights_only=False)
        logit_metric = audit.metric(actual['logits'], ref['logits'])
        loss_error = max(abs(float(a['lm loss'][0] / a['lm loss'][1] - b['lm loss'][0] / b['lm loss'][1]))
                         for a, b in zip(actual['losses'], ref['losses']))
        assert actual['grads'].keys() == ref['grads'].keys()
        grad_metrics = {key: audit.metric(value, ref['grads'][key]) for key, value in actual['grads'].items()}
        significant = {key: value for key, value in grad_metrics.items() if value['reference_l2'] >= 1e-3}
        worst = max(significant, key=lambda key: significant[key]['relative_l2'])
        assert logit_metric['relative_l2'] < (1e-5 if 'fp32' in name else .02), (name, logit_metric)
        assert loss_error < (2e-5 if 'fp32' in name else .005), (name, loss_error)
        assert all(m['finite'] for m in grad_metrics.values())
        assert significant[worst]['relative_l2'] < (1e-4 if 'fp32' in name else .05), (name, worst, significant[worst])
        results[name] = dict(logits=logit_metric, max_loss_abs=loss_error, worst_gradient=worst,
                             worst_gradient_metric=significant[worst], gradients=grad_metrics)
    (OUT / 'model-alignment.json').write_text(json.dumps(results, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(exist_ok=True, parents=True)
    if dist.get_rank() == 0:
        audit.prepare_fixture()
    dist.barrier()
    layouts = {'dense': (1, 1, False), 'tp2': (2, 1, False), 'tp2-sp': (2, 1, True),
               'pp2': (1, 2, False), 'tp2-pp2': (2, 2, False), 'tp2-sp-pp2': (2, 2, True),
               'tp2-sp-pp2-recompute': (2, 2, True), 'tp4-sp-pp2': (4, 2, True),
               'dense-fp32': (1, 1, False), 'tp2-sp-pp2-fp32': (2, 2, True)}
    names = args.cases or list(layouts)
    for name in names:
        run(name, *layouts[name], recompute='recompute' in name, fp32='fp32' in name)
    if dist.get_rank() == 0:
        compare(names)
        print('PASS ALL ALIGNMENT THRESHOLDS', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
