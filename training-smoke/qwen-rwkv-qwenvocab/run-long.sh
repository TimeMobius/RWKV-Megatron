#!/usr/bin/env bash
# Background entry point; records completion independently of this chat.
set -uo pipefail
distill_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$distill_repo_root"
bash training-smoke/qwen-rwkv-qwenvocab/launch.sh long
run_status=$?
exit_file="${RWKV_DISTILL_EXITCODE:-training-smoke/qwen-rwkv-qwenvocab/long-20261016-1536.exitcode}"
printf '%s\n' "$run_status" > "$exit_file"
exit "$run_status"
