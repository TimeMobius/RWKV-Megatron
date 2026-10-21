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
export CP_AUDIT_OUTPUT="$PWD/training-smoke/audit/cp-error-source-20261021/cp2-fp32-v2"
export CP_AUDIT_MASK=full
mkdir -p "$CP_AUDIT_OUTPUT"
torchrun --standalone --nproc-per-node 8 \
 training-smoke/audit/cp-error-source-20261021/capture_fp32_v2.py \
 --rwkv-config training-smoke/audit/cp-equivalence-20261021/config-cp2.yaml \
 > "$CP_AUDIT_OUTPUT/run.log" 2>&1
