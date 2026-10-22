#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius_py312
set -u
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$PWD/training-smoke/cache/inductor-py312"
cp_validation_case="$1"
cp_validation_gpus="$2"
export CP_VALIDATION_OUT="$PWD/training-smoke/audit/cp-validation-20261021/$cp_validation_case"
mkdir -p "$CP_VALIDATION_OUT"
python - "training-smoke/audit/cp-validation-20261021/config-$cp_validation_case.yaml" <<'CHECK_CONFIG'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
if config_path.exists():
    config = yaml.safe_load(config_path.read_text())
    if config.get('save') and not config.get('load'):
        checkpoint_dir = Path(config['save'])
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise SystemExit(f'Checkpoint directory is nonempty; use a new experiment directory: {checkpoint_dir}')
CHECK_CONFIG
torchrun --standalone --nproc-per-node "$cp_validation_gpus" \
 "${CP_VALIDATION_ENTRY:-training-smoke/audit/cp-validation-20261021/observe.py}" \
 --rwkv-config "training-smoke/audit/cp-validation-20261021/config-$cp_validation_case.yaml" \
 > "$CP_VALIDATION_OUT/run.log" 2>&1
