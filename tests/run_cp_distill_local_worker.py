"""Run the production entrypoint, recording first-step logits at CP boundaries."""

import atexit
import hashlib
import os
from pathlib import Path
import runpy

import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.training import get_args

ROWS = {}
ORIGINAL = GPTModel.forward


def forward(self, input_ids, *args, **kwargs):
    result = ORIGINAL(self, input_ids, *args, **kwargs)
    if self.post_process and self.training and get_args().curr_iteration == 0:
        # Keep all vocabulary logits at a small set of positions, including
        # both sides of every CP2/CP4 boundary. Never gather a full 4096xV batch.
        length = get_args().seq_length
        positions = sorted({0, 63, 64, length - 2, length - 1}
                           | {i * length // 4 + delta for i in (1, 2, 3) for delta in (-1, 0)})
        assert result.shape[:2] == input_ids.shape
        width = result.shape[1]
        start = ps.get_context_parallel_rank() * width
        selected = result.detach().new_zeros(result.shape[0], len(positions), result.shape[-1])
        for column, index in enumerate(positions):
            if start <= index < start + width:
                selected[:, column] = result[:, index - start].detach()
        tokens = input_ids.detach()
        if ps.get_context_parallel_world_size() > 1:
            dist.all_reduce(selected, group=ps.get_context_parallel_group())
            contexts = [torch.empty_like(tokens) for _ in range(ps.get_context_parallel_world_size())]
            dist.all_gather(contexts, tokens, group=ps.get_context_parallel_group())
            tokens = torch.cat(contexts, dim=1)
        if ps.get_tensor_model_parallel_world_size() > 1:
            shards = [torch.empty_like(selected) for _ in range(ps.get_tensor_model_parallel_world_size())]
            dist.all_gather(shards, selected, group=ps.get_tensor_model_parallel_group())
            selected = torch.cat(shards, dim=-1)
        if ps.get_context_parallel_rank() == 0 and ps.get_tensor_model_parallel_rank() == 0:
            for token, logits in zip(tokens.cpu(), selected.cpu()):
                key = hashlib.sha256(token.contiguous().numpy().tobytes()).hexdigest()
                assert key not in ROWS, "First-step samples must be distinct"
                ROWS[key] = logits
    return result


def finish():
    if ROWS:
        path = Path(os.environ["RWKV_CP_LOGITS_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        torch.save(ROWS, path / f"first-logits-rank{dist.get_rank()}.pt")


if __name__ == "__main__":
    GPTModel.forward = forward
    atexit.register(finish)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "pretrain_rwkv7.py"), run_name="__main__")
