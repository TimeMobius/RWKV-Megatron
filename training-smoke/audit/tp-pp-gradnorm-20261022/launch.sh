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
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/training-smoke/cache/inductor-py312"
export TPPP_DIAG_OUT="$PWD/training-smoke/audit/tp-pp-gradnorm-20261022/$1"
mkdir -p "$TPPP_DIAG_OUT"
exec torchrun --master_port "${TPPP_PORT:-51549}" --nproc-per-node "${TPPP_NPROC:-8}" --nnodes 1 \
 training-smoke/audit/tp-pp-gradnorm-20261022/capture.py --rwkv-config "$2"
