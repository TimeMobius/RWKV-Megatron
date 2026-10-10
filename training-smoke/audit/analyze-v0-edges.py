"""Check causal evidence for layer-zero BF16 gradient addition grouping."""
import json
from pathlib import Path

import torch

from check import metric
from v0_edges import summarize

PATH = Path(__file__).resolve().parent
BASE = 'full-bf16-fla-vs-fla-aligned-shared-pointops-te-final-carrier-chain-te-norms-v0-edge-audit'
CONFIRM = 'full-bf16-fla-vs-fla-all-gradients-aligned-shared-pointops-te-final-carrier-chain-te-norms-v0-confirm-split-value'


def main():
    baseline_stem=BASE+'-shared-value' if (PATH/(BASE+'-shared-value.json')).exists() else BASE
    baseline = json.loads((PATH / (baseline_stem + '.json')).read_text())
    old = json.loads((PATH / BASE.replace('-v0-edge-audit', '-new-replay-core')).with_suffix('.json').read_text())
    assert baseline['gradient_trace'] == old['gradient_trace']
    assert baseline['gradient_by_parameter'] == old['gradient_by_parameter']
    edges = torch.load(PATH / (baseline_stem + '-v0-edges.pth'), weights_only=True)
    a, b = edges['actual'], edges['reference']
    parts = lambda case: [r['gradient'] for r in case['records'] if r['kind'] == 'source-edge']
    pa, pb = parts(a), parts(b)
    assert len(pa) == 3 and len(pb) == 2
    assert torch.equal(pa[0], pb[0])
    assert torch.equal(pa[1] + pa[2], pb[1])
    assert torch.equal((pa[0] + pa[1]) + pa[2], a['total'])
    assert torch.equal(pa[0] + (pa[1] + pa[2]), b['total'])
    lerps = lambda case: [r['gradient'] for r in case['records'] if r['kind'] == 'lerp-end']
    la, lb = lerps(a), lerps(b)
    assert len(la) == len(lb) == 31
    assert all(torch.equal(x, y) for x, y in zip(la, lb))
    summary = {name: summarize(case) for name, case in edges.items()}
    assert summary['actual']['future_matches_reverse_layer_bf16_sum']
    # FP64 sums of fixed, already-rounded incoming gradients localize addition
    # error. They are not an independent full-model mathematical oracle.
    exact33 = sum(p.double() for p in la + pa[1:])
    rounded = exact33.to(a['total'].dtype)
    fp32_sum = torch.zeros_like(a['total'], dtype=torch.float32)
    for part in la + pa[1:]:
        fp32_sum = fp32_sum + part.float()
    assert torch.equal(fp32_sum.to(a['total'].dtype), rounded)
    result = {'observer_matches_previous_uninstrumented_run': True,
              'later_layers_with_exact_v0_branch_gradients': 31,
              'future_gradient_exact': True, 'local_two_branch_sum_exact': True,
              'both_totals_reconstructed_bit_exact_by_addition_grouping': True,
              'baseline_v0_difference': metric(a['total'], b['total']),
              'accumulation': summary,
              'fixed_33_branch_sum': {'actual_vs_fp64': metric(a['total'], exact33),
                                      'reference_vs_fp64': metric(b['total'], exact33),
                                      'fp32_sum_vs_fp64': metric(fp32_sum, exact33),
                                      'fp32_then_bf16_vs_fp64': metric(fp32_sum.to(a['total'].dtype), exact33),
                                      'one_final_bf16_round_vs_fp64': metric(rounded, exact33)},
              'runtime': baseline['runtime']}
    confirm = json.loads((PATH / (CONFIRM + '.json')).read_text())
    assert confirm['gradient_scope'] == 'all'
    assert len(confirm['gradient_by_parameter']) == 1059
    assert confirm['loss_delta'] == 0 and confirm['logits']['max_abs'] == 0
    assert all(x['max_abs'] == 0 for x in confirm['trace'].values())
    assert confirm['first_value_gradient']['max_abs'] == 0
    assert all(x['max_abs'] == 0 for x in confirm['gradient_by_parameter'].values())
    result['split_view_reference_control'] = {
        'forward_captures_exact': len(confirm['trace']), 'v0_gradient_exact': True,
        'parameter_gradients_checked': len(confirm['gradient_by_parameter']),
        'parameter_gradients_exact': sum(x['max_abs'] == 0 for x in confirm['gradient_by_parameter'].values()),
        'max_parameter_relative_l2': confirm['max_relative_gradient'],
        'remaining_differences': {k:v for k,v in confirm['gradient_by_parameter'].items() if v['max_abs'] != 0}}
    small_base=BASE.replace('full-', 'small-', 1)
    small_confirm=CONFIRM.replace('full-', 'small-', 1).replace('-all-gradients','')
    small=torch.load(PATH/(small_base+'-v0-edges.pth'),weights_only=True)
    sa,sb=parts(small['actual']),parts(small['reference'])
    assert len(sa)==3 and len(sb)==2
    assert torch.equal(sa[0],sb[0]) and torch.equal(sa[1]+sa[2],sb[1])
    assert torch.equal((sa[0]+sa[1])+sa[2],small['actual']['total'])
    assert torch.equal(sa[0]+(sa[1]+sa[2]),small['reference']['total'])
    small_result=json.loads((PATH/(small_confirm+'.json')).read_text())
    assert small_result['first_value_gradient']['max_abs']==0
    assert all(x['max_abs']==0 for x in small_result['gradient_by_parameter'].values())
    result['cropped_batch2_control']={'layers':4,'hidden':256,'sequence':128,'batch':2,
        'baseline_v0_difference':metric(small['actual']['total'],small['reference']['total']),
        'addition_grouping_reconstructs_both_totals':True,
        'aligned_parameter_gradients_exact':len(small_result['gradient_by_parameter'])}
    (PATH / 'v0-accumulation-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'runtime'}, indent=2))


if __name__ == '__main__':
    main()
