"""Targeted SP boundaries and observed selective/saved-activation behavior."""
import argparse
import importlib.util
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.training import global_vars
from rwkv_megatron.rwkv7_layer_specs import RWKV7TokenShiftSequenceParallel, RWKV7TimeMix, RWKV7ChannelMix

PATH=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_distributed_audit',PATH/'distributed.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def shift_check():
    rank=dist.get_rank();size=dist.get_world_size()
    ps.initialize_model_parallel(tensor_model_parallel_size=size)
    results=[]
    for local_seq in (1,2,32):
        for batch in (1,3):
            full=torch.arange(size*local_seq*batch*8,device='cuda',dtype=torch.float32).reshape(size*local_seq,batch,8)
            local=full.chunk(size,dim=0)[rank].clone().requires_grad_()
            actual=RWKV7TokenShiftSequenceParallel.apply(local,0)
            expected=torch.cat((torch.zeros_like(full[:1]),full[:-1]),dim=0).chunk(size,dim=0)[rank]
            assert torch.equal(actual,expected),(rank,local_seq,batch,'forward')
            weights=(full+1).square()
            actual.backward(weights.chunk(size,dim=0)[rank])
            expected_grad=torch.cat((weights[1:],torch.zeros_like(weights[:1])),dim=0).chunk(size,dim=0)[rank]
            assert torch.equal(local.grad,expected_grad),(rank,local_seq,batch,'backward')
            results.append(dict(local_seq=local_seq,batch=batch,forward_exact=True,backward_exact=True))
    if rank==0:(PATH/f'shift-tp{size}.json').write_text(json.dumps(results,indent=2)+'\n')
    ps.destroy_model_parallel()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True,choices=('shift','core_attn','mlp','layernorm','none','saved','singlemicro'))
    parser.add_argument('--fp32',action='store_true')
    parser.add_argument('--microbatches',type=int,default=1)
    args=parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    torch.backends.cuda.matmul.allow_tf32=False
    if args.variant=='shift':shift_check()
    else:
        audit.OUT=PATH/f"extra-{args.variant}{'-fp32' if args.fp32 else ''}-mb{args.microbatches}";audit.OUT.mkdir(exist_ok=True)
        audit.prepare_fixture()
        original=audit.config_for
        def config(*a,**kw):
            cfg=original(*a,**kw)
            if args.variant in ('core_attn','mlp','layernorm'):
                cfg.recompute_granularity='selective';cfg.recompute_method=None;cfg.recompute_num_layers=None;cfg.recompute_modules=[args.variant]
            if args.variant=='saved':cfg.distribute_saved_activations=True
            return cfg
        audit.config_for=config
        audit.DIMS['microbatches']=args.microbatches
        if args.variant=='singlemicro':
            audit.run_case('dense',1,1)
            # Fresh process groups aren't repeated enough here to cause resource exhaustion.
            audit.run_case('pp2',1,2)
        elif args.variant=='saved':
            audit.run_case('dense',1,1)
            audit.run_case('tp2-pp2',2,2)
            audit.run_case('tp2-pp2-recompute',2,2,recompute=True)
        else:
            counts={}
            original_core=audit.recurrence
            from rwkv_megatron.rwkv7_layer_specs import WKV7
            forward=WKV7.forward
            def capture(self,*a,**kw):
                key=str(self.layer_number) if hasattr(self,'layer_number') else str(id(self))
                row=counts.setdefault(key,{'no_grad':0,'with_grad':0})
                row['with_grad' if torch.is_grad_enabled() else 'no_grad']+=1
                return forward(self,*a,**kw)
            WKV7.forward=capture
            mlp_forward=RWKV7ChannelMix.forward
            def capture_mlp(self,*a,**kw):
                key='mlp-'+str(id(self))
                row=counts.setdefault(key,{'no_grad':0,'with_grad':0})
                row['with_grad' if torch.is_grad_enabled() else 'no_grad']+=1
                return mlp_forward(self,*a,**kw)
            RWKV7ChannelMix.forward=capture_mlp
            audit.run_case('tp2-sp-pp2-fp32-naive' if args.fp32 else 'tp2-sp-pp2',2,2,naive=args.fp32)
            (audit.OUT/f'rank{dist.get_rank()}-core-calls.json').write_text(json.dumps(counts,indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
