"""Reproducible real-weight/data validation configurations; no production defaults changed."""
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = yaml.safe_load((ROOT/'configs/config-rwkv7-2p7b-qwen-cp2-sp-smoke.yaml').read_text())


def case(name, *, cp=2, pp=2, sp=False, seq=4096, batch=2, steps=3, **extra):
    c = copy.deepcopy(BASE)
    c['model_parallel'].update(context_parallel_size=cp, pipeline_model_parallel_size=pp, sequence_parallel=sp)
    c.update(seq_length=seq, global_batch_size=batch, eval_global_batch_size=batch,
             train_iters=steps, eval_interval=steps, num_workers=0, distributed_timeout_minutes=5,
             data_cache_path=str(HERE/f'data-cache-seq{seq}'), **extra)
    (HERE/f'config-{name}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    return c


case('tp2-pp1-cp2', pp=1)
case('tp2-pp1-cp4-8k', cp=4, pp=1, seq=8192)
case('tp2-pp1-cp4-8k-measured', cp=4, pp=1, seq=8192)
case('tp2-pp1-cp4-sp-16k', cp=4, pp=1, sp=True, seq=16384)
case('tp2-pp2-cp2-wkv', steps=1, eval_iters=0)
for cp in (1, 4):
    case(f'capture-cp{cp}-8k', cp=cp, pp=1, seq=8192, steps=1, eval_iters=0)
case('capture-ordinary-cp2', sp=True, batch=4, steps=1, eval_iters=0)
for sp in (False, True):
    name = f'resume-sp{int(sp)}'
    ckpt = HERE/'checkpoints'/name
    c = case(name+'-continuous', sp=sp, steps=4, save=str(ckpt), save_interval=2)
    c['eval_interval'] = 2
    (HERE/f'config-{name}-continuous.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    if sp:
        regression = copy.deepcopy(c)
        regression['save'] = None
        (HERE/'config-ordinary-regression-sp1.yaml').write_text(yaml.safe_dump(regression, sort_keys=False))
    c.update(load=str(ckpt), ckpt_step=2, save=None)
    (HERE/f'config-{name}-restored.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
for kind in ('fp8', 'muon', 'fp8-muon', 'fp8-delayed'):
    c = case(kind, sp=True, seq=4096, batch=4, steps=3)
    if kind.startswith('fp8'):
        c['transformer_impl'] = 'transformer_engine'
        c['language_model']['fp8'] = 'hybrid'
        c['language_model']['fp8_recipe'] = 'delayed' if kind.endswith('delayed') else 'tensorwise'
    if 'muon' in kind:
        c.update(optimizer='muon', use_layer_wise_param_layout=False, muon_momentum=0.95,
                 muon_nesterov=True, muon_split_qkv=False, muon_scale_mode='spectral',
                 muon_num_ns_steps=5, muon_coefficient_type='quintic', muon_fp32_matmul_prec='medium',
                 muon_tp_mode='duplicated', muon_use_syrk=False, muon_extra_scale_factor=1., muon_scalar_optimizer='adam')
    (HERE/f'config-{kind}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
