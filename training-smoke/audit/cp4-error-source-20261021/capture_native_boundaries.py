"""Diagnostic full-model capture with native serial state propagation only."""
import runpy
from pathlib import Path

from ablate_states import DPLR, native_backward, native_forward

ROOT = Path(__file__).resolve().parents[3]

DPLR.chunk_gated_delta_rule_fwd_h_pre_process = lambda **kw: native_forward(kw)
DPLR.chunk_gated_delta_rule_bwd_dhu_pre_process = lambda **kw: (native_backward(kw), None)

if __name__ == '__main__':
    runpy.run_path(str(ROOT/'training-smoke/audit/cp-equivalence-20261021/capture.py'),run_name='__main__')
