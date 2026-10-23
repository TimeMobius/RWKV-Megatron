"""Summarize/assert full-size frozen-teacher training alignment from stdout."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training-smoke/parallel-distill-20261023"
LAYOUTS = ("dense", "tp2", "tp2-sp", "pp2", "tp2-sp-pp2", "tp4-sp-pp2")


def summarize():
    result = {}
    for name in LAYOUTS:
        log = (OUT / f"frozen-{name}.log").read_text()
        lines = [
            line
            for line in log.splitlines()
            if re.search(r"iteration\s+[123]/\s+3\s+\|", line)
        ]
        assert len(lines) == 3, (name, len(lines))
        steps = []
        for line in lines:

            def number(key):
                return float(re.search(re.escape(key) + r":\s+([\d.Ee+-]+)", line)[1])

            assert (
                number("number of skipped iterations")
                == number("number of nan iterations")
                == 0
            )
            steps.append(
                {
                    key: number(key)
                    for key in (
                        "lm loss",
                        "distill/kl_loss",
                        "distill/student_ce_loss",
                        "grad norm",
                        "elapsed time per iteration (ms)",
                    )
                }
            )
        urls = re.findall(
            r"http://dian\.sea-group\.org:9098/mobius/rwkv-qwen-distill-2p7b/runs/[a-z0-9]+",
            log,
        )
        assert urls, name
        assert (
            "'submitted': 12, 'consumed': 12, 'completed': 12, 'failed': 0" in log
        ), name
        result[name] = dict(wandb=urls[-1], steps=steps)
    reference = result["dense"]["steps"][0]
    for name, info in result.items():
        errors = {
            key: abs(info["steps"][0][key] - reference[key]) / abs(reference[key])
            for key in (
                "lm loss",
                "distill/kl_loss",
                "distill/student_ce_loss",
                "grad norm",
            )
        }
        for key, error in errors.items():
            assert error < (0.05 if key == "grad norm" else 0.005), (name, key, error)
        info["first_step_relative_errors"] = errors
    (OUT / "full-size-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    summarize()
