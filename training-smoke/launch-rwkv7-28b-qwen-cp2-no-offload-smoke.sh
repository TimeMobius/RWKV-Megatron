#!/usr/bin/env bash
# Tested all-GPU 28B CP2 configuration: seq2048, 30 steps, no checkpoint/W&B.
set -euo pipefail
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
phase="cp2-no-offload-smoke-$(date +%Y%m%d-%H%M%S)"
config="$PWD/configs/config-rwkv7-28b-qwen3.8-0056-cp2-no-offload-smoke.yaml"
export NCCL_MAX_NCHANNELS=1 NCCL_MIN_NCHANNELS=1
export NCCL_BUFFSIZE=262144 NCCL_NVLS_ENABLE=0
exec bash training-smoke/audit/28b-oom-20261022/launch.sh "$phase" "$config"
