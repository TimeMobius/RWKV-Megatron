"""End-to-end diagnostic using only the CP summary FP32-v2 intervention."""
import os
import runpy
import sys
from pathlib import Path
import torch
import megatron.training.training as training

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'training-smoke/audit/cp-equivalence-20261021'))
from capture import audit_train_step
from ablate_boundary import CP, fp32_v2_variant


def capture_with_variant(*args, **kwargs):
    original=CP.pre_process_fwd_kernel_merged
    CP.pre_process_fwd_kernel_merged=fp32_v2_variant()
    try:
        return audit_train_step(*args,**kwargs)
    finally:
        CP.pre_process_fwd_kernel_merged=original


if __name__=='__main__':
    torch.backends.cuda.matmul.allow_tf32=False
    training.train_step=capture_with_variant
    sys.argv[0]=str(ROOT/'pretrain_rwkv7.py')
    runpy.run_path(str(ROOT/'pretrain_rwkv7.py'),run_name='__main__')
