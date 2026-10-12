"""Check the local-head carrier across three PP boundaries with TP2/PP4."""
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.training import global_vars
from runtime import report
import distributed as wrapper

if __name__=='__main__':
    audit=wrapper.audit
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    torch.backends.cuda.matmul.allow_tf32=False
    audit.prepare_fixture()
    audit.run_case('tp2-sp-pp4',2,4,sp=True,recompute=True)
    dist.destroy_process_group()
