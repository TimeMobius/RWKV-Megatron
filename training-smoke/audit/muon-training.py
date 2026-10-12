"""Observe production Muon parameter routing and reuse fixed-eval capture."""

import json
import os
import runpy
from pathlib import Path

import megatron.core.optimizer as optimizer_module
from megatron.core.optimizer.layer_wise_optimizer import is_managed_by_layer_wise_optimizer

ROOT = Path(__file__).resolve().parents[2]
original_get_param_groups = optimizer_module._get_param_groups


def captured_param_groups(model_chunks, config, *args, **kwargs):
    groups = original_get_param_groups(model_chunks, config, *args, **kwargs)
    if config.optimizer != 'muon':
        return groups
    names = {id(param): name for chunk in model_chunks for name, param in chunk.named_parameters()}
    rows = []
    for group in groups:
        opt_name = group.get('optimizer', config.optimizer)
        for param in group['params']:
            name = names[id(param)]
            expected = 'muon' if is_managed_by_layer_wise_optimizer(param) else config.muon_scalar_optimizer
            assert opt_name == expected, (name, opt_name, expected)
            if name.endswith('mix_coeff_input') or getattr(param, 'is_embedding_or_output_parameter', False):
                assert opt_name == 'adam', (name, opt_name)
            if name.endswith(('weight_down', 'weight_up', 'linear_key.weight', 'linear_fc1.weight')):
                assert opt_name == 'muon', (name, opt_name)
            rows.append({'name': name, 'shape': list(param.shape), 'optimizer': opt_name,
                         'elements': param.numel()})
    output = ROOT / 'training-smoke/audit' / f"muon-routing-{os.environ['MUON_AUDIT_TAG']}.json"
    output.write_text(json.dumps({'parameters': rows}, indent=2) + '\n')
    return groups


if __name__ == '__main__':
    optimizer_module._get_param_groups = captured_param_groups
    runpy.run_path(str(ROOT / 'training-smoke/audit/fixed-eval-training.py'), run_name='__main__')
