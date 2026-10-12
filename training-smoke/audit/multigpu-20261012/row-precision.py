"""Audit-only FP32 partial outputs on row-parallel GEMMs; no production edit."""
import importlib.util
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.training import global_vars

PATH=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_distributed_audit',PATH/'distributed.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class WideOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,weight):
        ctx.save_for_backward(input,weight)
        shape=input.shape
        return torch.mm(input.reshape(-1,shape[-1]),weight.t(),out_dtype=torch.float32).reshape(*shape[:-1],weight.shape[0])

    @staticmethod
    def backward(ctx,grad):
        input,weight=ctx.saved_tensors
        g=grad.to(input.dtype).reshape(-1,weight.shape[0]).contiguous()
        x=input.reshape(-1,input.shape[-1])
        return torch.mm(g,weight).reshape_as(input),torch.mm(g.t(),x)


def forward_impl(input,weight,bias=None,**kwargs):
    assert bias is None and not kwargs['sequence_parallel'] and not kwargs['allreduce_dgrad']
    assert not kwargs['gradient_accumulation_fusion']
    return WideOutput.apply(input,weight)


original_forward=RowParallelLinear.forward


def row_forward(self,*args,**kwargs):
    self._forward_impl=forward_impl
    output,bias=original_forward(self,*args,**kwargs)
    return output.to(args[0].dtype),bias


if __name__=='__main__':
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    torch.backends.cuda.matmul.allow_tf32=False
    RowParallelLinear.forward=row_forward
    audit.FULL=True;audit.MINIPILE=True
    audit.DIMS=dict(layers=32,hidden=2560,heads=40,ffn=10240,vocab=65536,sequence=128,batch=1,microbatches=1)
    audit.run_case('tp2-row-wide-output',2,1)
    dist.destroy_process_group()
