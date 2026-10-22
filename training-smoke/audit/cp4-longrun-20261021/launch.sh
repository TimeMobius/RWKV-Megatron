#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius_py312
set -u
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/training-smoke/cache/inductor-py312"
exec python training-smoke/audit/cp4-longrun-20261021/supervise.py
