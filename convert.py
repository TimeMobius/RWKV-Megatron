import os

import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

CHECKPOINT_DIR = "/mnt/data/Outs/rwkv7-megatron/ckpt/iter_0000439"
TORCH_SAVE_CHECKPOINT_DIR = "/mnt/data/Outs/rwkv7-megatron/torch_save_checkpoint.pth"

# convert dcp model to torch.save (assumes checkpoint was generated as above)
dcp_to_torch_save(CHECKPOINT_DIR, TORCH_SAVE_CHECKPOINT_DIR)

# converts the torch.save model back to DCP
# dcp_to_torch_save(TORCH_SAVE_CHECKPOINT_DIR, f"{CHECKPOINT_DIR}_new")