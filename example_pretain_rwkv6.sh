#!/bin/bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=24
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc-per-node 8 --nnodes 1 pretrain_rwkv6.py --yaml-cfg rwkv6_config.yaml | tee output_wandb.log