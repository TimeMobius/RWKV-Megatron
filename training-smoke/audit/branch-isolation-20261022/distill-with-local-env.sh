#!/usr/bin/env bash
# Run a command with this worktree's dependencies in train_mobius_py312.
set -eo pipefail
source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius_py312
set -u
distill_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$distill_repo_root"
export PYTHONPATH="$distill_repo_root:$distill_repo_root/3rdparty/Megatron-LM-0.11.0:$distill_repo_root/3rdparty/flash-linear-attention-260115"
if [[ $# -eq 0 ]]; then
  python - <<'PY'
import importlib.util
import json
from pathlib import Path

root = Path.cwd().resolve()
origins = {}
for name in ("fla", "megatron.core"):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, name
    path = Path(spec.origin).resolve()
    assert path.is_relative_to(root), (name, str(path))
    origins[name] = str(path)
print(json.dumps(dict(worktree=str(root), imports=origins), indent=2))
PY
else
  exec "$@"
fi
