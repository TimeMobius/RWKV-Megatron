"""Production launch/pool/async/training with a deterministic teacher, no HTTP.

Every W&B run is tagged frozen-teacher to distinguish it from online Qwen.
"""

import os
import sys
from pathlib import Path

import yaml

from tools.launch_distillation import default_training_command, launch

if __name__ == "__main__":
    os.environ["WANDB_TAGS"] = "distill,parallel-validation,frozen-teacher"
    command = None
    if os.environ.get("RWKV_CP_LOGITS_DIR"):
        command = default_training_command(yaml.safe_load(Path(sys.argv[1]).read_text()))
        command[-1] = str(Path(__file__).with_name("run_cp_distill_local_worker.py").resolve())
    sys.exit(
        launch(
            sys.argv[1],
            command=command,
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
