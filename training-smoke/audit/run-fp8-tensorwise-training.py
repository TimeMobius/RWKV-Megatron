"""Same-version FP8 tensorwise resume trial, with copied configs and new checkpoints."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    results = {}
    for phase in ("save", "resume", "continuous"):
        ckpt = (
            ROOT
            / "training-smoke/checkpoints"
            / (
                "single-continuous-fp8-tensorwise-ffn"
                if phase == "continuous"
                else "single-resume-fp8-tensorwise-ffn"
            )
        )
        if phase != "resume" and ckpt.exists():
            raise RuntimeError(f"Use a new checkpoint directory: {ckpt}")
        command = [
            "bash",
            "training-smoke/audit/launch-training-fixed-eval.sh",
            f"single-training-{phase}-fp8-tensorwise-ffn",
            f"training-smoke/config-3b-{phase}-fp8-tensorwise-ffn.yaml",
        ]
        env = dict(
            os.environ, FIXED_EVAL_CAPTURE=f"fp8-tensorwise-ffn-{phase}-batches.json"
        )
        proc = subprocess.run(command, env=env, cwd=ROOT)
        results[phase] = {"command": command, "exit_code": proc.returncode}
        (ROOT / "training-smoke/audit/fp8-tensorwise-training-exits.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        if proc.returncode:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
