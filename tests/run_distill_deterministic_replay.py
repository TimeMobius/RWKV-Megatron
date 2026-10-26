"""Exercise the production deterministic scan policy with independent autotuning."""

import json
import os
import subprocess
import sys

import yaml
from compare_numerics_replays import compare
from run_distill_numerics_cases import CKPTS, OUT, ROOT


def check_precision(cases):
    import torch
    from compare_dp_distill import canonical

    sys.path.insert(0, str(ROOT / "training-smoke/audit"))
    from distributed import metric

    dense = torch.load(
        ROOT / "training-smoke/dp2-distill-20261023/dense-bf16/step1-rank0.pt",
        weights_only=False,
    )
    reference = canonical([dense], "grads", 1, 1)
    report = {}
    for case in cases:
        rows = [
            torch.load(OUT / case / f"step1-rank{rank}.pt", weights_only=False)
            for rank in range(8)
        ]
        gradients = canonical(rows, "grads", 2, 2)
        metrics = {
            key: metric(value, reference[key]) for key, value in gradients.items()
        }
        significant = {
            key: value
            for key, value in metrics.items()
            if value["reference_l2"] >= 1e-3
        }
        key = max(significant, key=lambda key: significant[key]["relative_l2"])
        logits = {}
        for row in rows:
            if row["tp"] == 0 and row["pp"] == 1:
                logits.update(row["logits"])
        keys = sorted(dense["logits"])
        lm = metric(
            torch.stack([logits[key] for key in keys]),
            torch.stack([dense["logits"][key] for key in keys]),
        )
        assert lm["finite"] and lm["relative_l2"] < 0.02
        assert all(value["finite"] for value in metrics.values())
        assert significant[key]["relative_l2"] < 0.05
        report[case] = dict(
            logits=lm,
            worst_significant_parameter=key,
            worst_significant_gradient=significant[key],
        )
    (OUT / "deterministic-precision-20261026.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main():
    cases = []
    for sp in (False, True):
        name = "combined-" + ("sp-" if sp else "") + "detfix-20261026"
        cases.append(name)
        for resume in (False, True):
            case = name + ("-resume" if resume else "")
            folder = OUT / case
            folder.mkdir(parents=True, exist_ok=True)
            cfg = yaml.safe_load((OUT / "combined-original-none.yaml").read_text())
            cfg.update(
                deterministic_mode=True, save=str(CKPTS / case), load=None, eval_iters=1
            )
            cfg["model_parallel"]["sequence_parallel"] = sp
            if resume:
                source = CKPTS / (case + "-source")
                source.mkdir(parents=True, exist_ok=True)
                link = source / "iter_0000002"
                if not link.exists():
                    link.symlink_to(
                        CKPTS / name / "iter_0000002", target_is_directory=True
                    )
                (source / "latest_checkpointed_iteration.txt").write_text("2")
                cfg["load"] = str(source)
            config = OUT / f"{case}.yaml"
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            env = dict(
                os.environ,
                RWKV_DP_AUDIT_DIR=str(folder),
                RWKV_DP_FP32="0",
                DIAG_TRACE_LEVEL="none",
                FLA_CACHE_RESULTS="0",
            )
            # All unpinned kernels independently benchmark their valid configs.
            for key in (
                "DIAG_DGK_LAYOUT",
                "DIAG_DGK_CAPTURE",
                "DIAG_ALL_WKV_INPUTS",
                "DIAG_BF16_FULL_REDUCTION",
                "DIAG_ROW_FP32",
                "DIAG_REWARM_CHECKPOINT",
            ):
                env.pop(key, None)
            with (OUT / f"{case}.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc-per-node=8",
                        str(ROOT / "tests/trace_distill_numerics.py"),
                        "--rwkv-config",
                        str(config),
                    ],
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                )
            assert result.returncode == 0, (case, result.returncode)
            print("COMPLETE DETERMINISTIC", case, flush=True)
        compare([name])
        report = json.loads((OUT / "replay-controls.json").read_text())[name]
        assert report["bit_identical"], report
        scan_name = (
            "fla.ops.generalized_delta_rule.dplr.chunk_A_bwd.chunk_dplr_bwd_dgk_kernel"
        )
        for suffix in ("", "-resume"):
            for rank in range(8):
                kernels = json.loads(
                    (
                        OUT / (name + suffix) / f"kernels-step3-rank{rank}.json"
                    ).read_text()
                )
                assert kernels[scan_name] == {
                    "fixed": "BK: 32, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None"
                }
    full = json.loads((OUT / "replay-controls.json").read_text())
    (OUT / "deterministic-replay-20261026.json").write_text(
        json.dumps({name: full[name] for name in cases}, indent=2) + "\n"
    )
    check_precision(cases)


if __name__ == "__main__":
    main()
