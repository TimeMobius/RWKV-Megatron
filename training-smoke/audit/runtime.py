"""Verify actual source and native-library origins inside the training interpreter."""
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton
import transformer_engine
import transformer_engine.pytorch
import fla
import megatron.core
import fused_layer_norm_cuda
import fused_weight_gradient_mlp_cuda
from fla.ops.rwkv7 import chunk_rwkv7
from megatron.core.pipeline_parallel import schedules

ROOT = Path(__file__).resolve().parents[2]
ENV = Path('/mnt/data/anaconda3/envs/train_mobius_py312')


def report():
    sources = {}
    for name, module in [('megatron', megatron.core), ('fla', fla),
                         ('schedule', schedules), ('torch', torch), ('triton', triton),
                         ('transformer_engine', transformer_engine),
                         ('apex_layernorm', fused_layer_norm_cuda),
                         ('apex_wgrad', fused_weight_gradient_mlp_cuda)]:
        path = Path(module.__file__).resolve()
        expected = ROOT / '3rdparty' if name in ('megatron', 'fla', 'schedule') else ENV
        assert path.is_relative_to(expected), (name, path, expected)
        sources[name] = str(path)
    assert Path(sys.executable).is_relative_to(ENV), sys.executable
    function_path = Path(inspect.getsourcefile(inspect.unwrap(chunk_rwkv7))).resolve()
    assert function_path.is_relative_to(ROOT / '3rdparty/flash-linear-attention-260115')
    return dict(executable=sys.executable, prefix=sys.prefix, python=sys.version,
                torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
                transformer_engine=transformer_engine.__version__, sources=sources,
                rwkv_kernel=str(function_path), modules_from_expected_locations=True,
                submodule_commits={name:subprocess.check_output(['git','-C',str(ROOT / path),'rev-parse','HEAD'],text=True).strip()
                                  for name,path in [('megatron','3rdparty/Megatron-LM-0.11.0'),
                                                    ('fla','3rdparty/flash-linear-attention-260115')]},
                ld_library_path=os.environ.get('LD_LIBRARY_PATH'),
                source_sha256={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ['rwkv_megatron/rwkv7_layer_specs.py',
                                            'rwkv_megatron/rwkv7_transformer_layer.py',
                                            'rwkv_megatron/normalization.py',
                                            'rwkv_megatron/rwkv7_model.py']})


if __name__ == '__main__':
    result=report()
    (ROOT/'training-smoke/audit/runtime.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
