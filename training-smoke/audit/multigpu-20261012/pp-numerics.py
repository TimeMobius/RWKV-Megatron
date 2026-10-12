"""Forward-only replay locating the first PP BF16 primitive/input difference."""
import argparse
import importlib.util
import os
import shutil
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.training import global_vars

PATH=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_distributed_audit',PATH/'distributed.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
core_inputs={}
original_model=audit.RWKV7Model


def model(*a,**kw):
    module=original_model(*a,**kw)
    for layer in module.decoder.layers:
        index=layer.self_attention.layer_number-1
        def capture(m,inputs,index=index):
            for key,t in zip(('r','k','v','w','a','b'),inputs):core_inputs[f'{index}.core-{key}']=t.detach().cpu()
        layer.self_attention.core_attention.register_forward_pre_hook(capture)
    return module


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pp',type=int,default=1);parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    torch.backends.cuda.matmul.allow_tf32=False
    audit.OUT=PATH/'pp-numerics';audit.OUT.mkdir(exist_ok=True)
    shutil.copyfile(PATH/'minipile-tokens.pth',audit.OUT/'minipile-tokens.pth') if dist.get_rank()==0 else None
    dist.barrier()
    audit.FULL=True;audit.MINIPILE=True
    audit.DIMS=dict(layers=32,hidden=2560,heads=40,ffn=10240,vocab=65536,sequence=128,batch=1,microbatches=1)
    audit.RWKV7Model=model;audit.select_gradient=lambda *a:False
    schedule=audit.get_forward_backward_func
    def forward_only():
        fn=schedule()
        def call(**kwargs):kwargs['forward_only']=True;return fn(**kwargs)
        return call
    audit.get_forward_backward_func=forward_only
    audit.run_case(args.tag,1,args.pp)
    torch.save(core_inputs,audit.OUT/f'{args.tag}-rank{dist.get_rank()}-core-inputs.pth')
    dist.destroy_process_group()
