#!/usr/bin/env bash
set -eo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius_py312
set -u
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/mnt/data/anaconda3/envs/train_mobius_py312
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib/python3.12/site-packages/nvidia/cublas/lib:$CUDA_HOME/lib/python3.12/site-packages/nvidia/cusparse/lib:$CUDA_HOME/lib/python3.12/site-packages/nvidia/cusolver/lib:$CUDA_HOME/lib/python3.12/site-packages/nvidia/cudnn/lib:$CUDA_HOME/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo_dir:$repo_dir/3rdparty/Megatron-LM-0.11.0:$repo_dir/3rdparty/flash-linear-attention-260115${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
exec python tools/evaluate_english_benchmarks.py "$@"
