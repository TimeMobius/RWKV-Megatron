# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Transport native DPLR boundary states without an affine approximation."""

import torch
import torch.distributed as dist

from fla.ops.cp import FLACPContext


def receive_recurrent_state(context: FLACPContext, state: torch.Tensor, backward: bool = False) -> torch.Tensor:
    """Receive only the sequence continued across the current rank boundary."""
    rank = dist.get_rank(context.group)
    if backward:
        if not context.is_last_rank:
            dist.recv(state[-1], src=dist.get_global_rank(context.group, rank + 1), group=context.group)
    elif not context.is_first_rank:
        dist.recv(state[0], src=dist.get_global_rank(context.group, rank - 1), group=context.group)
    return state


def send_recurrent_state(context: FLACPContext, state: torch.Tensor, backward: bool = False) -> None:
    """Send the native state or state gradient to its adjacent CP rank."""
    rank = dist.get_rank(context.group)
    if backward:
        if not context.is_first_rank:
            dist.send(state[0].contiguous(), dst=dist.get_global_rank(context.group, rank - 1), group=context.group)
    elif not context.is_last_rank:
        dist.send(state[-1].contiguous(), dst=dist.get_global_rank(context.group, rank + 1), group=context.group)
