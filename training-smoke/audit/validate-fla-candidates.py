"""Check candidate evidence without enabling candidates in production."""
import json
from pathlib import Path

PATH=Path(__file__).resolve().parent


def read(name):return json.loads((PATH/(name+'.json')).read_text())


def main():
    summary=read('fla-candidate-summary')
    assert not summary['production_changed'] and not summary['environment_changed']
    assert not summary['fla_source_changed']
    exact=summary['full_model_vs_independent_fp32']['wkv-save']['vs_current']
    assert exact['loss_exact'] and exact['logits_exact'] and exact['v0_gradient_exact']
    assert exact['parameter_gradients_exact']==1059 and exact['forward_captures_exact']==546
    for mode,row in summary['full_model_vs_independent_fp32'].items():
        assert len(row['gradient_by_parameter'])==1059
        assert all(m['finite'] for m in row['gradient_by_parameter'].values())
    # Preserve the negative result: combining two locally more accurate ops
    # worsened this sample's full-model gradient error; don't call it a repair.
    model=summary['full_model_vs_independent_fp32']
    assert model['norm-correction']['aggregate_parameter_gradient_relative_l2']>model['current']['aggregate_parameter_gradient_relative_l2']
    for row in summary['local_operators']['wkv']['accuracy'].values():
        assert row['output']['finite'] and row['output']['relative_l2']<.007
        assert all(m['finite'] and m['relative_l2']<.008 for m in row['gradient'])
    formula=read('small-fp32-naive-vs-blinkdl-simplified-all-gradients-fla-candidate-fp32-formula-blinkdl-source-candidate-norm-correction')
    assert formula['loss_delta']==0 and formula['logits']['relative_l2']<1e-5
    assert len(formula['gradient_by_parameter'])==135
    assert formula['max_relative_gradient']<1e-5
    features=read('fla-features-results')
    assert not features['tilelang_installed']
    assert all(not row['shape_eligible'] for row in features['tilelang_eligibility_only_not_execution'] if row['batch']==1)
    assert features['features']['output_correction']['exists_in_original_muon']
    assert not features['features']['tilelang_backend']['exists_in_original_muon']
    assert all(row['exit_code']==0 for row in read('fla-candidate-exits').values())
    assert read('fla-candidate-first-exits')['model-benchmark']['exit_code']==1
    print('FLA CANDIDATE EVIDENCE VERIFIED; no production defaults or dependencies changed')


if __name__=='__main__':main()
