#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/mnt/data/anaconda3/envs/train_mobius_py312
export LD_LIBRARY_PATH=/mnt/data/anaconda3/envs/train_mobius_py312/lib:/mnt/data/anaconda3/envs/train_mobius_py312/targets/x86_64-linux/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cublas/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cusparse/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cusolver/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cudnn/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius_py312
distill_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$distill_repo_root"
export PYTHONPATH="$PWD:$PWD/3rdparty/Megatron-LM-0.11.0:$PWD/3rdparty/flash-linear-attention-260115"
mkdir -p training-smoke/qwen-rwkv-qwenvocab

teacher_env="${RWKV_TEACHER_ENV_FILE:-$HOME/.config/rwkv-distill/teacher.env}"
if [[ -f "$teacher_env" ]]; then
  set -a
  source "$teacher_env"
  set +a
fi

if [[ "${1:-train}" == precision ]]; then
  torchrun --standalone --nproc-per-node=1 training-smoke/qwen-rwkv-qwenvocab/precision.py \
    2>&1 | tee training-smoke/qwen-rwkv-qwenvocab/precision.log
  exit 0
fi

if [[ "${1:-train}" == test-async ]]; then
  python -m pytest -q tests/test_online_teacher.py tests/test_teacher_prefetch.py tests/test_fixed_eval_samples.py
  exit 0
fi

if [[ "${1:-train}" == async-precision ]]; then
  torchrun --standalone --nproc-per-node=1 training-smoke/qwen-rwkv-qwenvocab/async_precision.py \
    2>&1 | tee training-smoke/qwen-rwkv-qwenvocab/async-precision.log
  exit 0
fi

if [[ "${1:-train}" == jsonl-test ]]; then
  config="${RWKV_DISTILL_CONFIG:-configs/config-rwkv7-qwen-distill-jsonl-minipile-test.yaml}"
  run_log="${RWKV_DISTILL_LOG:-training-smoke/qwen-rwkv-qwenvocab/jsonl-minipile-test.log}"
  WANDB_MODE="${WANDB_MODE:-online}" torchrun --standalone --nproc-per-node=1 pretrain_rwkv7.py \
    --rwkv-config "$config" \
    2>&1 | tee "$run_log"
  exit 0
fi

config="${RWKV_DISTILL_CONFIG:-configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-single.yaml}"
run_log="${RWKV_DISTILL_LOG:-training-smoke/qwen-rwkv-qwenvocab/single.log}"
if [[ "${1:-train}" == long ]]; then
  config="${RWKV_DISTILL_CONFIG:-configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-long.yaml}"
  run_log="${RWKV_DISTILL_LOG:-training-smoke/qwen-rwkv-qwenvocab/long-20261016-1536.log}"
fi

WANDB_MODE="${WANDB_MODE:-online}" torchrun --standalone --nproc-per-node=1 pretrain_rwkv7.py \
  --rwkv-config "$config" \
  2>&1 | tee "$run_log"
