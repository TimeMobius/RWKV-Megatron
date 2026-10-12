"""Check checkpoint forward vs its own replay on identical saved inputs."""
import argparse
import importlib.util
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.training import global_vars
from analyze import metric

PATH=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_distributed_audit',PATH/'distributed.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
original_model=audit.RWKV7Model
records={}


def model(*a,**kw):
    module=original_model(*a,**kw)
    for layer in module.decoder.layers:
        index=layer.self_attention.layer_number-1
        def capture(m,inputs,output,index=index):
            row=dict(grad_enabled=torch.is_grad_enabled())
            for key,t in zip(('r','k','v','w','a','b'),inputs):row[key]=t.detach().cpu()
            row['output']=output.detach().cpu()
            records.setdefault(str(index),[]).append(row)
        layer.self_attention.core_attention.register_forward_hook(capture)
    return module


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--eager-key',action='store_true');parser.add_argument('--tp',type=int,default=1);parser.add_argument('--pp',type=int,default=1);parser.add_argument('--sp',action='store_true');args=parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    torch.backends.cuda.matmul.allow_tf32=False
    tag=('eager' if args.eager_key else 'compiled')+(f'-tp{args.tp}-pp{args.pp}' if args.tp>1 or args.pp>1 else '')
    audit.OUT=PATH/f'recompute-replay-{tag}';audit.OUT.mkdir(exist_ok=True)
    if dist.get_rank()==0:shutil.copyfile(PATH/'minipile-tokens.pth',audit.OUT/'minipile-tokens.pth')
    dist.barrier()
    audit.FULL=True;audit.MINIPILE=True
    audit.DIMS=dict(layers=32,hidden=2560,heads=40,ffn=10240,vocab=65536,sequence=128,batch=1,microbatches=1)
    original_config=audit.config_for
    def config(*a,**kw):
        cfg=original_config(*a,**kw);cfg.rwkv_key_fusion=not args.eager_key;return cfg
    audit.config_for=config
    audit.RWKV7Model=model;audit.select_gradient=lambda *a:False
    audit.run_case('recompute',args.tp,args.pp,sp=args.sp,recompute=True)
    result={}
    for layer,rows in records.items():
        assert len(rows)==2,(layer,len(rows))
        assert [r['grad_enabled'] for r in rows]==[False,True],(layer,rows)
        result[layer]={key:metric(rows[1][key],rows[0][key]) for key in ('r','k','v','w','a','b','output')}
    (PATH/f"recompute-replay-{tag}{'-rank'+str(dist.get_rank()) if dist.get_world_size()>1 else ''}.json").write_text(json.dumps(result,indent=2)+'\n')
    print('REPLAY INEXACT',[(layer,key,m['relative_l2']) for layer,values in result.items() for key,m in values.items() if not m['exact']],flush=True)
    dist.destroy_process_group()
