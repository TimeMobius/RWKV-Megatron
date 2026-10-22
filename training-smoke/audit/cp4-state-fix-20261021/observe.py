"""Reuse production training observations; opt into old affine mode only for timing."""
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if os.environ.get('CP_FIX_AFFINE_BASELINE') == '1':
    import rwkv_megatron.rwkv7_layer_specs as specs
    original_replace = specs.replace
    def affine_context(context, **kwargs):
        kwargs['dplr_state_mode'] = 'affine'
        return original_replace(context, **kwargs)
    specs.replace = affine_context

if __name__ == '__main__':
    runpy.run_path(str(ROOT/'training-smoke/audit/cp-validation-20261021/observe.py'),run_name='__main__')
