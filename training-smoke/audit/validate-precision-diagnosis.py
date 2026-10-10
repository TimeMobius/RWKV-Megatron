"""Check measured precision improvements without declaring BF16 globally aligned."""
import json
from pathlib import Path

PATH=Path(__file__).resolve().parent


def read(stem):return json.loads((PATH/(stem+'.json')).read_text())


def main():
    operators=read('operator-precision-results')
    assert operators['fp32_formula_checks_passed'] and len(operators['results'])==11
    for name,row in operators['results'].items():
        for label in ('current_bf16','official_bf16','current_fp32'):
            assert all(m['finite'] for group in ('output','gradient') for m in row[label][group].values())
        assert max(m['relative_l2'] for m in row['current_fp32']['output'].values())<1e-5
        assert max(m['relative_l2'] for m in row['current_fp32']['gradient'].values())<1e-3
        if name.endswith('.ffn'):assert row['current_vs_official_forward']['exact']
    summary=read('precision-diagnosis-summary')
    value=summary['v0_real_backward']
    assert value['all_forward_captures_exact']==546 and value['loss_exact']
    assert value['later_layer_gradient_branches_exact']==31 and value['local_two_gradient_branches_exact']
    assert value['fp32_source_sum_relative_l2']<1e-6
    assert value['fp32_then_bf16_projection_gradient_relative_l2']<value['baseline_sum_relative_l2']
    assert value['projection_gradient_matches_one_final_bf16_round']
    assert len(value['changed_parameters_only'])==7
    comparisons=summary['vs_independent_fp32']
    assert all(c['parameter_gradients']==1059 for c in comparisons.values())
    assert comparisons['key_fp32']['all_parameter_gradient_relative_l2']<comparisons['production_bf16']['all_parameter_gradient_relative_l2']
    checkpoint=read('v0-fp32-recompute-summary')
    assert checkpoint['forward_matches_baseline'] and checkpoint['loss_matches_baseline']
    assert checkpoint['checkpoint_vs_non_checkpoint_parameter_gradients_exact']==135
    assert all(row['exit_code']==0 for row in read('precision-probe-exits').values())
    exits=read('precision-extra-exits')
    assert exits['operator-initial']['exit_code']==1
    assert all(row['exit_code']==0 for name,row in exits.items() if name!='operator-initial')
    print('PRECISION EVIDENCE VERIFIED; production unchanged; full BF16 accuracy/parallel training not certified')


if __name__=='__main__':main()
