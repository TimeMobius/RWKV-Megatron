#!/usr/bin/env bash
# 28B, TP2/ordinary PP4/CP1/SP, seq4096, 30 steps, no checkpoint/W&B.
set -euo pipefail
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
phase="tp2pp4cp1-no-offload-smoke-$(date +%Y%m%d-%H%M%S)"
config="$PWD/configs/config-rwkv7-28b-qwen3.8-0056-tp2-pp4-no-offload-smoke.yaml"
exec bash training-smoke/audit/28b-oom-20261022/launch.sh "$phase" "$config"
