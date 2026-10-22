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
cp_fix_case="$1"
cp_fix_gpus="${2:-8}"
cp_fix_dir="$PWD/training-smoke/audit/cp4-state-fix-20261021"
export CP_VALIDATION_OUT="$cp_fix_dir/$cp_fix_case"
mkdir -p "$CP_VALIDATION_OUT"
torchrun --standalone --nproc-per-node "$cp_fix_gpus" \
 "${CP_FIX_ENTRY:-training-smoke/audit/cp4-state-fix-20261021/observe.py}" \
 --rwkv-config "$cp_fix_dir/config-$cp_fix_case.yaml" > "$CP_VALIDATION_OUT/run.log" 2>&1
