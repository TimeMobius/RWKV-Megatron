"""Launch the production teacher pool with a deterministic local test teacher."""

import runpy
from pathlib import Path

import torch

from rwkv_megatron.online_teacher import OnlineTeacherClient, TeacherTopK


def frozen_get_logits(self, tokens, **kwargs):
    offsets = torch.arange(self.top_k)
    ids = (
        tokens.detach().cpu()[..., None] + offsets * 31
    ) % self.get_effective_vocab_size()
    logp = (-offsets.float() / 10).expand_as(ids).clone()
    return TeacherTopK(ids, logp)


if __name__ == "__main__":
    OnlineTeacherClient.get_logits = frozen_get_logits
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/teacher_pool_server.py"),
        run_name="__main__",
    )
