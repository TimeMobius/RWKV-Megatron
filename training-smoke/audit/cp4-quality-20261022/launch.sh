#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius_py312
set -u
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR="$PWD/training-smoke/cache/inductor-py312"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
quality_out="$PWD/training-smoke/audit/cp4-quality-20261022"
case "$1" in
  initial|final)
    export CUDA_VISIBLE_DEVICES="${QUALITY_GPUS:-0}"
    exec torchrun --standalone --nproc-per-node=1 "$quality_out/evaluate.py" --model "$1"
    ;;
  initial-expanded|final-expanded)
    exec torchrun --standalone --nproc-per-node=8 \
      training-smoke/audit/cp4-longrun-20261021/entry.py \
      --rwkv-config "$quality_out/config-$1.yaml"
    ;;
  check-export)
    export CUDA_VISIBLE_DEVICES="${QUALITY_GPUS:-2,3}"
    exec torchrun --standalone --nproc-per-node=2 "$quality_out/check_export.py"
    ;;
  *)
    echo "Usage: bash launch.sh {initial|final|initial-expanded|final-expanded|check-export}" >&2
    exit 2
    ;;
esac
