import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

dcp_to_torch_save("/mnt/data/Outs/rwkv7-megatron/ckpt/iter_0002000", "/mnt/data/Outs/rwkv7-megatron/torch")
