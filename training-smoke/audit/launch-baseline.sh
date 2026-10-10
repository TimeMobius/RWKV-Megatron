#!/usr/bin/env bash
set -eo pipefail

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR=/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/training-smoke/cache/inductor-py311
export TRITON_CACHE_DIR=/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/training-smoke/cache/triton-py311
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/mnt/data/anaconda3/envs/train_mobius
export LD_LIBRARY_PATH=/mnt/data/anaconda3/envs/train_mobius/lib:/mnt/data/anaconda3/envs/train_mobius/targets/x86_64-linux/lib:/mnt/data/anaconda3/envs/train_mobius/lib/python3.11/site-packages/nvidia/cublas/lib:/mnt/data/anaconda3/envs/train_mobius/lib/python3.11/site-packages/nvidia/cusparse/lib:/mnt/data/anaconda3/envs/train_mobius/lib/python3.11/site-packages/nvidia/cusolver/lib:/mnt/data/anaconda3/envs/train_mobius/lib/python3.11/site-packages/nvidia/cudnn/lib:/mnt/data/anaconda3/envs/train_mobius/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius
set -u

cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/training-smoke/baseline-replay
cp /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/training-smoke/audit/old_model.py audit_old.py
exec python -m torch.distributed.run --master_port 51285 --nproc_per_node 1 audit_old.py
