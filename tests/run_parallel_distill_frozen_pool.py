"""Launch the production teacher pool with a deterministic local test teacher."""

import runpy
import argparse
from pathlib import Path

import torch
import yaml

from rwkv_megatron.online_teacher import OnlineTeacherClient, TeacherTopK

EXPECTED_LENGTH = None


def frozen_get_logits(self, tokens, **kwargs):
    if EXPECTED_LENGTH is not None:
        assert tokens.shape[1] == EXPECTED_LENGTH, "CP teacher must receive the full sequence"
    offsets = torch.arange(self.top_k)
    ids = (
        tokens.detach().cpu()[..., None] + offsets * 31
    ) % self.get_effective_vocab_size()
    logp = (-offsets.float() / 10).expand_as(ids).clone()
    return TeacherTopK(ids, logp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rwkv-config", required=True)
    args, _ = parser.parse_known_args()
    EXPECTED_LENGTH = yaml.safe_load(Path(args.rwkv_config).read_text()).get("seq_length")
    OnlineTeacherClient.get_logits = frozen_get_logits
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/teacher_pool_server.py"),
        run_name="__main__",
    )
