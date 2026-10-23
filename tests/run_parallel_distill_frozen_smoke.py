"""Production launch/pool/async/training with a deterministic teacher, no HTTP.

Every W&B run is tagged frozen-teacher to distinguish it from online Qwen.
"""

import os
import sys
from pathlib import Path

from tools.launch_distillation import launch

if __name__ == "__main__":
    os.environ["WANDB_TAGS"] = "distill,parallel-validation,frozen-teacher"
    sys.exit(
        launch(
            sys.argv[1],
            pool_command=[
                sys.executable,
                "-u",
                str(
                    Path(__file__)
                    .with_name("run_parallel_distill_frozen_pool.py")
                    .resolve()
                ),
                "--rwkv-config",
                str(Path(sys.argv[1]).resolve()),
            ],
        )
    )
