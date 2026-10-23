"""Deterministic teacher injection into the actual training entrypoint.

Only test scaffolding: never contacts HTTP. Use RWKV_TEST_TEACHER_FAIL=1 to
verify torchrun tears down PP/TP peers on a teacher error.
"""

import os
import runpy
from pathlib import Path

import torch

from rwkv_megatron.online_teacher import OnlineTeacherClient, TeacherTopK


def frozen_get_logits(self, tokens, **kwargs):
    if os.environ.get("RWKV_TEST_TEACHER_FAIL") == "1":
        raise TimeoutError("injected teacher outage")
    ids = (tokens.detach().cpu()[..., None] + torch.tensor([0, 31, 67, 98])) % 119
    logp = -torch.tensor([0.2, 1.1, 2.3, 4.0]).expand_as(ids).clone()
    return TeacherTopK(ids, logp)


if __name__ == "__main__":
    OnlineTeacherClient.get_logits = frozen_get_logits
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "pretrain_rwkv7.py"),
        run_name="__main__",
    )
