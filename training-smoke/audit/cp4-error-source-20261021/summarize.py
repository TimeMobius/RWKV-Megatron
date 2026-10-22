"""Check the CP4 boundary ablations and independent affine/precision controls."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYERS = {'0','4','13','23'}
MODES = ('default','native_forward','native_backward','native_both')


def main():
    ablations = []
    for tp in (0,1):
        for rank in range(4):
            data = json.loads((HERE/f'ablation-tp{tp}-rank{rank}.json').read_text())
            assert set(data) == LAYERS
            for layer,variants in data.items():
                ablations.append(dict(tp=tp,cp=rank,layer=int(layer),variants=variants))
    summary = dict(layers=[0,4,13,23],total_heads=36,sequence_length=8192,
                   rank_layer_records=len(ablations),ablations={})
    for mode in MODES:
        rows = [r['variants'][mode] for r in ablations]
        assert all(r['output']['finite'] and all(g['finite'] for g in r['gradients'].values()) for r in rows)
        summary['ablations'][mode] = dict(
            max_output_relative_l2=max(r['output']['relative_l2'] for r in rows),
            max_input_gradient_relative_l2=max(g['relative_l2'] for r in rows for g in r['gradients'].values()),
            all_outputs_exact=all(r['output']['exact'] for r in rows),
            all_gradients_exact=all(g['exact'] for r in rows for g in r['gradients'].values()))
    assert abs(summary['ablations']['default']['max_output_relative_l2']-0.00674422224983573) < 1e-12
    assert abs(summary['ablations']['default']['max_input_gradient_relative_l2']-0.012060284614562988) < 1e-12
    assert summary['ablations']['native_both']['all_outputs_exact']
    assert summary['ablations']['native_both']['all_gradients_exact']
    summary['boundary_pattern'] = {}
    for rank in range(4):
        rows = [r['variants']['default'] for r in ablations if r['cp'] == rank]
        summary['boundary_pattern'][str(rank)] = dict(
            max_output_relative_l2=max(r['output']['relative_l2'] for r in rows),
            max_forward_boundary_relative_l2=max(r['boundaries']['forward_cp_vs_reference']['relative_l2'] for r in rows),
            max_backward_boundary_relative_l2=max(r['boundaries']['backward_cp_vs_reference']['relative_l2'] for r in rows))
    summary['contracts'] = {}
    for precision in ('default','tf32x3','ieee'):
        records = []
        for tp in (0,1):
            for rank in range(4):
                data = json.loads((HERE/f'contract-{precision}-tp{tp}-rank{rank}.json').read_text())
                assert set(data) == {'4','13'}
                records.extend(data.values())
        assert all(r['all_gather_hashes_exact'] for r in records)
        result = dict(all_gather_hashes_exact=True,records=len(records))
        for direction in ('forward','backward'):
            rows = [r[direction] for r in records if direction in r]
            assert all(r['summary_dtype'] == r['initial_dtype'] == 'torch.float32' for r in rows)
            assert all(r['production']['zero_state_vs_summary']['exact'] for r in rows)
            result[direction] = dict(
                matrix_summary_max_relative_l2=max(r['matrix_summary_vs_fp64']['relative_l2'] for r in rows),
                zero_state_summaries_exact=True,variants={})
            for variant in ('production','ieee_bf16_state','float32_control'):
                values = [r[variant] for r in rows if variant in r]
                if values:
                    result[direction]['variants'][variant] = dict(
                        affine_residual_max_relative_l2=max(v['affine_residual']['relative_l2'] for v in values),
                        midpoint_residual_max_relative_l2=max(v['midpoint_residual']['relative_l2'] for v in values))
            assert result[direction]['variants']['float32_control']['affine_residual_max_relative_l2'] < 1e-5
        summary['contracts'][precision] = result
    baseline = json.loads((HERE.parent/'cp-validation-20261021/comparison-cp4-8k.json').read_text())
    native = json.loads((HERE/'comparison-native.json').read_text())
    assert native['passed'] and native['gradient_aggregate']['all_finite']
    assert all(v['max_abs'] == 0 for v in native['forward'].values())
    summary['full_model'] = dict(
        gradient_tensors=native['gradient_aggregate']['tensor_count'],
        baseline_gradient_relative_l2=baseline['gradient_aggregate']['relative_l2'],
        native_boundary_gradient_relative_l2=native['gradient_aggregate']['relative_l2'],
        captured_forward_exact=True,all_gradients_finite=True,
        capture='all token CE, full head input, 64 sampled vocabulary logits per TP rank; two microbatches')
    summary['production_changes'] = False
    summary['limits'] = [
        'Controlled WKV replay covers 4 real layers/all 36 heads, including both original worst-error layers.',
        'Affine controls cover layers 4 and 13; IEEE/FP32 variants are diagnostics, not production changes.',
        'Serial native state propagation identifies the source; no parallel training performance claim.',
        'Residual full-model 0.0169% gradient difference has not been traced element by element here.']
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'contracts'},indent=2))


if __name__ == '__main__':
    main()
