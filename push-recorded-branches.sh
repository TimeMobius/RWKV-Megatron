#!/usr/bin/env bash
# Publish dependency branches before their parent repository branches.
set -euo pipefail
cp_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
distill_repo="$(dirname -- "$cp_repo")/RWKV-Megatron-distill"

git -C "$cp_repo/3rdparty/Megatron-LM-0.11.0" push -u origin codex/rwkv7-cp
git -C "$cp_repo/3rdparty/flash-linear-attention-260115" push -u origin codex/rwkv7-cp
git -C "$cp_repo" push -u org codex/rwkv7-cp

git -C "$distill_repo/3rdparty/Megatron-LM-0.11.0" push -u origin codex/qwen-rwkv28b-distill
git -C "$distill_repo/3rdparty/flash-linear-attention-260115" push -u origin codex/qwen-rwkv28b-distill
git -C "$distill_repo" push -u org codex/qwen-rwkv28b-distill
