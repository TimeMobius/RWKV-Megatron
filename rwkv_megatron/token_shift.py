"""Logical sequence neighbors for TP sequence shards inside contiguous CP shards."""

from functools import lru_cache

import torch
import torch.distributed as dist


@lru_cache(maxsize=32)
def tp_cp_shift_neighbors(group, cp_rank, tp_rank, tp_size, device):
    """Map (CP, TP) token order to group peers, independently of physical rank order.

    Every member calls this once before its first shift; later calls reuse the
    group-local mapping. The combined group contains only one PP/DP replica.
    """
    coordinates = torch.tensor([cp_rank, tp_rank], dtype=torch.int64, device=device)
    gathered = [
        torch.empty_like(coordinates) for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(gathered, coordinates, group=group)
    peers = {tuple(value.tolist()): index for index, value in enumerate(gathered)}
    logical_rank = cp_rank * tp_size + tp_rank
    previous = peers.get(divmod(logical_rank - 1, tp_size))
    following = peers.get(divmod(logical_rank + 1, tp_size))
    return previous, following
