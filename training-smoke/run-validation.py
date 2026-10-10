#!/usr/bin/env python3
"""Run training validations sequentially and record actual launcher exit codes."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "baseline-today": ["bash", "training-smoke/launch-baseline-replay.sh", "baseline-today"],
    "fla-only-today": ["bash", "training-smoke/launch-fla-only-replay.sh", "fla-only-today"],
    **{
        phase: ["bash", "training-smoke/launch-baseline-replay.sh", phase]
        for phase in ("baseline-warm", "baseline-warm-repeat")
    },
    **{
        phase: ["bash", "training-smoke/launch-fla-only-replay.sh", phase]
        for phase in ("fla-only-warm", "fla-only-warm-repeat")
    },
    **{
        phase: ["bash", "training-smoke/launch-3b.sh", phase]
        for phase in ("megatron-upstream-30-final", "megatron-upstream-30-final-repeat")
    },
    **{
        phase: ["bash", "training-smoke/launch-3b.sh", phase, f"training-smoke/config-3b-{phase}.yaml"]
        for phase in ("checkpoint-save", "checkpoint-resume", "checkpoint-continuous")
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", choices=list(RUNS))
    args = parser.parse_args()
    status_file = ROOT / "training-smoke/validation-exits.json"
    statuses = json.loads(status_file.read_text()) if status_file.exists() else {}
    for phase in args.runs:
        started = datetime.now(timezone.utc).isoformat()
        print(f"Starting {phase}", flush=True)
        with (ROOT / f"training-smoke/logs/{phase}.console").open("w") as console:
            result = subprocess.run(RUNS[phase], cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
        statuses[phase] = {
            "command": RUNS[phase], "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": result.returncode,
        }
        status_file.write_text(json.dumps(statuses, indent=2) + "\n")
        print(f"Finished {phase}: exit {result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
