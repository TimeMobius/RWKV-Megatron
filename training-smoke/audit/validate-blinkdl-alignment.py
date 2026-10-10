"""Validate official-source formula evidence and report BF16 differences."""
import hashlib
import json
import math
from pathlib import Path

PATH=Path(__file__).resolve().parent


def load(stem):
    return json.loads((PATH/(stem+'.json')).read_text())


def summarize(case):
    gradients=case['gradient_by_parameter']
    denominator=sum(v['reference_l2']**2 for v in gradients.values())
    aggregate=math.sqrt(sum((v['relative_l2']*v['reference_l2'])**2 for v in gradients.values())/denominator)
    return {'loss_megatron':case['loss_megatron'],'loss_reference':case['loss_native'],
            'loss_delta':case['loss_delta'],'logits_relative_l2':case['logits']['relative_l2'],
            'v0_gradient_relative_l2':case['first_value_gradient']['relative_l2'],
            'parameter_gradients':len(gradients),'aggregate_gradient_relative_l2':aggregate,
            'max_parameter_gradient_relative_l2':case['max_relative_gradient'],
            'worst_parameter':max(gradients,key=lambda k:gradients[k]['relative_l2']),
            'exact_parameter_gradients':sum(v['max_abs']==0 for v in gradients.values()),
            'forward_trace_count':len(case['trace']),
            'exact_forward_traces':sum(v['max_abs']==0 for v in case['trace'].values())}


def main():
    names={
        'full_fp32':'full-fp32-naive-vs-blinkdl-simplified-all-gradients-official-fp32-blinkdl-source',
        'small_fp32':'small-fp32-naive-vs-blinkdl-simplified-official-fp32-blinkdl-source',
        'full_bf16_official':'full-bf16-fla-vs-blinkdl-main-all-gradients-official-bf16-blinkdl-source',
        'small_bf16_official':'small-bf16-fla-vs-blinkdl-main-official-bf16-blinkdl-source',
        'full_direct_vfirst_diagnostic':'full-bf16-fla-vs-fla-all-gradients-aligned-shared-pointops-te-final-te-norms-official-direct-vfirst',
        'small_direct_vfirst_diagnostic':'small-bf16-fla-vs-fla-all-gradients-aligned-shared-pointops-te-final-te-norms-official-direct-vfirst',
        'full_bf16_torch_norms':'full-bf16-fla-vs-blinkdl-main-all-gradients-torch-norms-official-bf16-torch-norms-blinkdl-source',
    }
    cases={key:load(name) for key,name in names.items()}
    for name,case in cases.items():
        assert case['gradient_scope']=='all'
        assert len(case['gradient_by_parameter'])==(1059 if name.startswith('full') else 135)
        assert all(v['finite'] for v in case['gradient_by_parameter'].values())
        assert case['logits']['finite'] and case['first_value_gradient']['finite']
        if 'diagnostic' not in name:
            provenance=case['reference_provenance']
            assert provenance['exact_source_forward_bodies']
            assert provenance['v_first_transport']=='direct object reuse, no carrier'
            assert case['reference_profile']=='blinkdl'
            for filename,expected in provenance['source_sha256'].items():
                source=PATH.parent/'reference-repos/RWKV-LM/RWKV-v7/train_temp'/filename
                assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
        if name.endswith('fp32'):
            # Existing frozen full-model formula thresholds, not BF16 bit equality.
            assert abs(case['loss_delta'])<1e-5
            assert case['logits']['relative_l2']<1e-5
            assert case['max_relative_gradient']<1e-3
            assert case['first_value_gradient']['relative_l2']<1e-4
        if 'diagnostic' in name:
            assert case['loss_delta']==0 and case['logits']['max_abs']==0
            assert case['first_value_gradient']['max_abs']==0
            assert all(v['max_abs']==0 for v in case['gradient_by_parameter'].values())
            assert all(v['max_abs']==0 for v in case['trace'].values())
    kernels=load('blinkdl-kernel-results')
    assert [r['sequence'] for r in kernels]==[64,128]
    assert all(r['passed'] and not r['failures'] for r in kernels)
    assert all(r['thresholds']=={'output':0.007,'gradients':0.008} for r in kernels)
    exits=load('blinkdl-validation-exits')
    assert all(row['exit_code']==0 for row in exits.values())
    result={'cases':{k:summarize(v) for k,v in cases.items()},
            'formula_thresholds_passed':True,
            'same_primitives_direct_vfirst_all_gradients_exact':True,
            'actual_official_bf16_bit_exact':False,
            'actual_official_bf16_accuracy_alignment_certified':False,
            'kernel_frozen_thresholds_passed':True,
            'reference_provenance':cases['full_bf16_official']['reference_provenance'],
            'runtime':cases['full_bf16_official']['runtime'],
            'limitations':['One full 3B MiniPile prefix, seq128, batch1; cropped4-layer batch2.',
                           'Official source bodies and CUDA operators, eager execution; no full official trainer/JIT/DeepSpeed checkpoint test.',
                           'Plain CE throughout, excluding official L2Wrap.',
                           'FP32 simplified forward uses independent torch recurrence instead of demo CUDA.',
                           'Shared-primitive direct-vfirst diagnostic is not an official-kernel precision oracle.',
                           'No production code or environment change; no multi-GPU, FP8 or convergence certification.']}
    (PATH/'blinkdl-alignment-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['cases'],indent=2))


if __name__=='__main__':main()
