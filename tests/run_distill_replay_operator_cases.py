"""Run isolated historical checkpoint controls without overwriting old evidence."""

import argparse
import os

from run_distill_numerics_cases import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "workspace16",
            "workspace4096",
            "fullreduce",
            "eager-mix",
            "dgk-fixed",
            "dgk-historical",
        ],
        required=True,
    )
    parser.add_argument("--sp", action="store_true")
    args = parser.parse_args()
    for mode in args.modes:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        os.environ.pop("DIAG_BF16_FULL_REDUCTION", None)
        os.environ.pop("DIAG_DGK_LAYOUT", None)
        if mode.startswith("workspace"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
                ":16:8" if mode == "workspace16" else ":4096:8"
            )
        if mode == "fullreduce":
            os.environ["DIAG_BF16_FULL_REDUCTION"] = "1"
        if mode.startswith("dgk-"):
            os.environ["DIAG_DGK_LAYOUT"] = mode.removeprefix("dgk-")
        name = "combined-" + ("sp-" if args.sp else "") + f"original-{mode}-20261026"
        run(name, 2, 2, 2, sp=args.sp, steps=3)


if __name__ == "__main__":
    main()
