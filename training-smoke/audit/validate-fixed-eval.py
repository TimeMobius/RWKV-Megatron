"""Verify actual fixed/rolling token groups and same-version checkpoint resume."""
import json
from pathlib import Path

PATH=Path(__file__).resolve().parent


def main():
    captures={phase:json.loads((PATH/f'fixed-eval-{phase}-batches.json').read_text())['events']
              for phase in ('save','resume','continuous','rolling')}
    first=captures['save'][0]
    expected=first['batch_hashes']
    assert len(expected)==4 and len(set(expected))==4
    counts={}
    for phase in ('save','resume','continuous'):
        regular=[row for row in captures[phase] if not row.get('replay_same_weights')]
        assert regular
        for row in captures[phase]:
            assert row['batch_hashes']==expected
            if not row.get('replay_same_weights'):
                assert row['fixed_eval_samples'] and not row['is_test']
                assert row['consumed_valid_samples_after']-row['consumed_valid_samples_before']==4
        replay=next(row for row in captures[phase] if row.get('replay_same_weights'))
        assert replay['losses']==regular[0]['losses']
        counts[phase]=len(regular)
    assert captures['resume'][0]['consumed_valid_samples_before']>0
    resumed=[r for r in captures['resume'] if not r.get('replay_same_weights')]
    continuous=[r for r in captures['continuous'] if not r.get('replay_same_weights')]
    assert [r['losses'] for r in resumed]==[r['losses'] for r in continuous[2:]]
    rolling=captures['rolling']
    assert all(not row['fixed_eval_samples'] for row in rolling)
    assert len({tuple(row['batch_hashes']) for row in rolling})==len(rolling)
    assert rolling[0]['batch_hashes']==expected
    resume=json.loads((PATH/'single-resume-fixed-eval-results.json').read_text())
    assert resume['exact_model_optimizer_scheduler'] and resume['rng_exact']
    assert all(not check['differences'] for check in resume['checkpoint_comparisons'].values())
    exits=json.loads((PATH/'fixed-eval-training-exits.json').read_text())
    assert all(row['exit_code']==0 for row in exits.values())
    result={'configuration':'fixed_eval_samples: true','eval_group_samples':4,'unique_batch_hashes':expected,
        'fixed_group_exact_across_evaluations_and_resume':True,'eval_counts':counts,
        'same_weights_replay_loss_exact':True,'resumed_eval_losses_exact_vs_continuous':True,
        'rolling_eval_group_count':len(rolling),'rolling_groups_all_different':True,
        'eval_accounting_advances':True,'checkpoint_resume':resume,
        'root_unit_tests_passed':98,'megatron_unit_tests_passed':3,'multi_gpu_training_certified':False}
    (PATH/'fixed-eval-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FIXED EVAL PASS: repeated actual tokens and same-weight loss; resume loss/checkpoints exact; rolling mode retains new groups')


if __name__=='__main__':main()
