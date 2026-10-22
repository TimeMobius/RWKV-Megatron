"""Keep evaluation-only dataset indices identical to the training job's indices."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '3rdparty/Megatron-LM-0.11.0'))

import megatron.training.training as training  # noqa: E402

original_num_samples = training.get_train_valid_test_num_samples


def matched_num_samples():
    args = training.get_args()
    skip_train = args.skip_train
    try:
        args.skip_train = False
        return original_num_samples()
    finally:
        args.skip_train = skip_train


training.get_train_valid_test_num_samples = matched_num_samples
runpy.run_path(str(ROOT / 'pretrain_rwkv7.py'), run_name='__main__')
