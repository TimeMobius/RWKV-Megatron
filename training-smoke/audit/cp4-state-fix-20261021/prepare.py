"""New copies of established real-model CP test configurations."""
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'cp-validation-20261021'


def case(name, source, **updates):
    config = yaml.safe_load((BASE/f'config-{source}.yaml').read_text())
    config.update(updates)
    (HERE/f'config-{name}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))


case('strict-cp4-8k','tp2-pp1-cp4-8k-measured')
case('strict-cp2-4k','tp2-pp1-cp2')
case('strict-cp4-sp-16k','tp2-pp1-cp4-sp-16k')
case('capture-cp4-8k','capture-cp4-8k')
case('tp2-pp2-cp2-sp-mb2','ordinary-regression-sp1',train_iters=3,eval_interval=3,
     micro_batch_size=2,global_batch_size=4,eval_micro_batch_size=2,eval_global_batch_size=4)
for mode in ('affine','recurrent'):
    case(f'timing-{mode}','tp2-pp1-cp4-8k-measured',train_iters=8,eval_iters=0,
         eval_interval=1000000,log_throughput=True)
for sp in (0,1):
    for phase in ('continuous','restored'):
        case(f'resume-sp{sp}-{phase}',f'resume-sp{sp}-{phase}',
             save=str(HERE/'checkpoints'/f'sp{sp}') if phase=='continuous' else None,
             load=None if phase=='continuous' else str(HERE/'checkpoints'/f'sp{sp}'))
