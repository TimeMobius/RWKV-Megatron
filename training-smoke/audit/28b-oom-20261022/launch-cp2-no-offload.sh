#!/usr/bin/env bash
# Real 28B all-GPU Adam smoke test, CP2 with TE parameter remainders.
set -euo pipefail
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
phase="${1:-cp2-no-offload-2k}"
case "$phase" in
  cp2-no-offload-2k|cp2-no-offload-4k-recompute2) ;;
  *) printf '%s\n' 'Supported phases: cp2-no-offload-2k, cp2-no-offload-4k-recompute2' >&2; exit 2 ;;
esac
# Explicit values are part of the tested memory configuration.
export NCCL_MAX_NCHANNELS=1 NCCL_MIN_NCHANNELS=1
export NCCL_BUFFSIZE=262144 NCCL_NVLS_ENABLE=0
exec bash training-smoke/audit/28b-oom-20261022/launch.sh "$phase"
