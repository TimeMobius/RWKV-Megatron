"""Validate per-rank evidence and retain distinct functional/precision outcomes."""
import json
import math
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def training_case(name, ranks, steps):
    paths = sorted((HERE/name).glob('rank*.json'))
    assert len(paths) == ranks, (name, len(paths))
    rows = [json.loads(p.read_text()) for p in paths]
    cfg = yaml.safe_load((HERE/f'config-{name}.yaml').read_text())
    for row in rows:
        events = [e for e in row['events'] if e['kind'] == 'train']
        assert [e['iteration'] for e in events] == steps, name
        assert all(e['skipped'] == 0 and math.isfinite(e['grad_norm']) for e in events), name
        assert all(math.isfinite(v) for e in row['events'] for v in e['losses'].values()), name
    observed = next(row for row in rows if any(e['losses'] for e in row['events']))
    result = dict(
        ranks=ranks, topology={k:cfg['model_parallel'][k] for k in (
            'tensor_model_parallel_size','pipeline_model_parallel_size','context_parallel_size','sequence_parallel')},
        sequence_length=cfg['seq_length'], optimizer=cfg['optimizer'], steps=steps,
        train_losses=[e['losses']['lm loss'] for e in observed['events'] if e['kind'] == 'train'],
        validation_losses=[e['losses']['lm loss'] for e in observed['events'] if e['kind'] == 'eval'],
        max_allocated_gib=max(e.get('peak_allocated_gib',0) for row in rows for e in row['events']),
        all_finite=True, skipped_steps=0, functional_passed=True)
    if name.startswith('fp8'):
        for row in rows:
            assert row['fp8_dispatch']
            assert all(v['fp8'] == ('.mlp.' in k) for k,v in row['fp8_dispatch'].items())
            assert any('Float8' in k and 'grad=False' in k for k in row['gemm_operands'])
            assert any('Float8' in k and 'grad=True' in k for k in row['gemm_operands'])
            assert row['wkv_dtypes'] == ['torch.bfloat16']
        result['fp8_verified_on_every_rank'] = True
        result['fp8_scope'] = 'FFN only; real Float8 GEMM forward/backward, WKV BF16'
    if 'muon' in name:
        for row in rows:
            assert {r['optimizer'] for r in row['optimizer_routing']} == {'muon','adam'}
        result['routing_counts_including_cp_tp_replicas'] = dict(Counter(
            r['optimizer'] for row in rows for r in row['optimizer_routing']))
        result['muon_layout'] = 'use_layer_wise_param_layout=false; muon_tp_mode=duplicated'
    return result


def wkv_case(name, ranks):
    records = []
    for rank in range(ranks):
        row = json.loads((HERE/name/f'rank{rank}.json').read_text())
        assert set(row['wkv']) == {str(i) for i in range(24)}
        for layer, r in row['wkv'].items():
            assert r['heads'] == 18 and set(r['gradients']) == {'r','k','v','w','a','b'}
            assert r['output']['finite'] and all(v['finite'] for v in r['gradients'].values())
            records.append(dict(rank=rank, layer=int(layer), **r))
    worst_o = max(records,key=lambda r:r['output']['relative_l2'])
    worst_g = max((dict(rank=r['rank'],layer=r['layer'],input=k,**v)
                   for r in records for k,v in r['gradients'].items()),key=lambda r:r['relative_l2'])
    return dict(layers=24, total_heads=36, microbatches_replayed=1, batch_size=1,
                shard_records=len(records), all_finite=True,
                output_limit=0.001, input_gradient_limit=0.01,
                strict_passed=all(r['passed'] for r in records),
                failed_shard_records=sum(not r['passed'] for r in records),
                all_outputs_exact=all(r['output']['exact'] for r in records),
                all_input_gradients_exact=all(v['exact'] for r in records for v in r['gradients'].values()),
                worst_output=dict(rank=worst_o['rank'],layer=worst_o['layer'],**worst_o['output']),
                worst_input_gradient=worst_g)


if __name__ == '__main__':
    cases = [('tp2-pp1-cp2',4,[1,2,3]), ('tp2-pp1-cp4-8k-measured',8,[1,2,3]),
             ('tp2-pp1-cp4-sp-16k',8,[1,2,3]),
             *[(k,8,[1,2,3]) for k in ('fp8','fp8-delayed','muon','fp8-muon')],
             ('ordinary-regression-sp1',8,[1,2,3,4])]
    summary = dict(training={n:training_case(n,r,s) for n,r,s in cases},
                   cp2_wkv=wkv_case('tp2-pp1-cp2',4),
                   cp4_wkv=wkv_case('tp2-pp1-cp4-8k-measured',8),
                   resume={str(sp):json.loads((HERE/f'resume-sp{sp}-comparison.json').read_text())
                           for sp in (0,1)},
                   vpp='Deferred by user; experimental production changes withdrawn, no VPP claim',
                   limitations=['3/4-step functional checks, no long-run convergence certification',
                                'CP4 strict per-layer WKV numerical comparison FAILED',
                                'FP8 short-run execution verified, precision/convergence not certified',
                                'Checkpoint comparison uses Adam, torch format, unchanged topology'])
    comparison = json.loads((HERE/'comparison-cp4-8k.json').read_text())
    summary['cp4_model'] = dict(passed_existing_bf16_limits=comparison['passed'],
        gradient_aggregate=comparison['gradient_aggregate'], limits=comparison['limits'],
        max_token_ce_absolute=max(v['max_abs'] for k,v in comparison['forward'].items() if '-token_loss-' in k),
        max_sampled_logits_relative_l2=max(v['relative_l2'] for k,v in comparison['forward'].items() if '-logits-' in k))
    assert summary['cp2_wkv']['strict_passed']
    assert all(r['passed'] for r in summary['resume'].values())
    summary['ordinary_pp_regression'] = json.loads((HERE/'ordinary-regression-comparison.json').read_text())
    assert summary['ordinary_pp_regression']['passed']
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Functional cases validated; CP2 WKV exact; CP4 strict WKV failure retained.')
