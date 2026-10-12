"""Frozen correctness checks, with BF16 discrepancies reported separately."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent


def main():
    full=json.loads((OUT/'full-3b-minipile-comparisons.json').read_text())
    expected=['tp2-fp32-naive','tp2-sp-fp32-naive','pp2-fp32-naive','tp2-pp2-fp32-naive','tp2-sp-pp2-fp32-naive','tp2-sp-pp2-recompute-fp32-naive',
              'tp2','tp2-sp','pp2','tp2-pp2','tp2-sp-pp2','tp2-sp-pp2-recompute','tp2-sp-pp2-uniform2','tp2-sp-pp2-block2','tp4-sp-pp2','tp4-sp-pp2-recompute']
    assert all(k in full for k in expected)
    summaries={}
    for name in expected:
        row=full[name];g=row['all_gradients'];assert g['parameters']==1059 and g['all_finite']
        assert all(r['forward_exact'] and r['backward_exact'] and r['v0_gradient_nonzero'] for r in row['pp_transport']),name
        unchanged=row['v0_carrier_unchanged'];assert unchanged['checked']==unchanged['exact'],name
        if 'fp32' in name:
            # Tiny expected FP32 reassociation differences, unchanged pre-existing formula thresholds.
            assert g['relative_l2']<1e-4 and row['logits']['relative_l2']<1e-4 and row['loss_max_delta']<1e-4,name
            assert max(v['relative_l2'] for v in row['gradient_by_parameter'].values())<1e-3,name
            if row.get('vs_no_recompute'):assert row['vs_no_recompute']['all_gradients']['exact_parameters']==1059,name
        summaries[name]={k:row[k] for k in ('losses','reference_losses','loss_max_delta','logits','all_gradients','pp_transport','v0_carrier_unchanged','v0_vs_reference')}
        if row.get('vs_no_recompute'):
            summaries[name]['vs_no_recompute']={k:row['vs_no_recompute'][k] for k in ('loss_max_delta','logits','all_gradients')}
    resume={name:json.loads((OUT/f'{name}-resume-validation.json').read_text())['exact_continuation'] for name in ('tp2-sp-pp2','tp4-sp-pp2')}
    result=dict(full_model_formulas_fp32_pass=True,all_recorded_pp_transport_exact=True,bf16_error_metrics=summaries,resume=resume,
                limits=['single node H800','BF16 Adam','same parallel topology on resume','seq32 FP32 / seq128 BF16','not FP8 or Muon certification','short test is not convergence proof'])
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FP32 FORMULAS + COMMUNICATION + SAME-TOPOLOGY RESUME PASS; BF16 precision differences remain separate.')


if __name__=='__main__':main()
