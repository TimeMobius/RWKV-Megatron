"""Select the frozen source snapshot in a fresh diagnostic process."""
import runpy
import importlib.util
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent
old_dgk = '--old-dgk' in sys.argv
old_intra = '--old-intra' in sys.argv
for flag in ('--old-dgk', '--old-intra'):
    if flag in sys.argv:
        sys.argv.remove(flag)
if '--old-fla' in sys.argv:
    sys.argv.remove('--old-fla')
    sys.path.insert(0, str(PATH.parent / 'reference-repos/fla-79d12e8e'))
    import runtime
    import fla
    import torch
    # This source is intentionally outside production's import-origin check.
    # Record its actual path and frozen SHA instead of relabeling it production.
    runtime.report = lambda: dict(reference_fla=True,
        reference_fla_commit='79d12e8ec0ae72bc7f2189b4244d6532ce8c268c',
        source=fla.__file__, executable=sys.executable, torch=torch.__version__)
if old_dgk or old_intra:
    old_source = PATH.parent / 'reference-repos/fla-79d12e8e/fla/ops/generalized_delta_rule/dplr/chunk_A_bwd.py'
    spec = importlib.util.spec_from_file_location('fla_old_intra_diagnostic', old_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import fla.ops.generalized_delta_rule.dplr.chunk_A_bwd as current_intra
    import fla.ops.generalized_delta_rule.dplr.chunk as current_chunk
    if old_intra:
        current_chunk.chunk_dplr_bwd_dqk_intra = module.chunk_dplr_bwd_dqk_intra
    else:
        current_intra.chunk_dplr_bwd_dgk_kernel = module.chunk_dplr_bwd_dgk_kernel
    import runtime
    original_report = runtime.report
    runtime.report = lambda: {**original_report(), 'diagnostic_override':
        'old-intra' if old_intra else 'old-dgk', 'override_source': str(old_source)}
runpy.run_path(str(PATH / 'single.py'), run_name='__main__')
