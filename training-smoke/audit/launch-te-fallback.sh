#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD:$PWD/training-smoke/audit:${PYTHONPATH:-}"
export CUDA_HOME=/mnt/data/anaconda3/envs/train_mobius_py312
export LD_LIBRARY_PATH=/mnt/data/anaconda3/envs/train_mobius_py312/lib:/mnt/data/anaconda3/envs/train_mobius_py312/targets/x86_64-linux/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cublas/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cusparse/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cusolver/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cudnn/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
exec /mnt/data/anaconda3/envs/train_mobius_py312/bin/python -m torch.distributed.run --master_port 51287 --nproc_per_node 1 training-smoke/audit/te-fallback.py "$@"
