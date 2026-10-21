#!/usr/bin/env bash
# 8 GPUs: TP2 x PP2 x CP2, real 2.7B weights/data, 3 steps and validation.
set -euo pipefail

cp_smoke_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$cp_smoke_root/training-smoke/launch-rwkv7-28b-qwen-day1-cp2.sh" \
  "${1:-qwen-rwkv2p7b-cp2-smoke-$(date +%Y%m%d-%H%M%S)}" \
  "${2:-$cp_smoke_root/configs/config-rwkv7-2p7b-qwen-cp2-smoke.yaml}"
