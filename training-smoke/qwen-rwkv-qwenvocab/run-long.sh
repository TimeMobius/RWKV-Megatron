#!/usr/bin/env bash
# Background entry point; records completion independently of this chat.
set -uo pipefail
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
bash training-smoke/qwen-rwkv-qwenvocab/launch.sh long
run_status=$?
exit_file="${RWKV_DISTILL_EXITCODE:-training-smoke/qwen-rwkv-qwenvocab/long-20261016-1536.exitcode}"
printf '%s\n' "$run_status" > "$exit_file"
exit "$run_status"
