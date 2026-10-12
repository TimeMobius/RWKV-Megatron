"""Capture full-precision train/eval scalars and data hashes per rank, no math changes."""
import hashlib
import json
import os
import runpy
from pathlib import Path

import torch
import torch.distributed as dist
import megatron.training.training as training
from megatron.core import parallel_state as ps
from megatron.core.utils import unwrap_model
from megatron.core.rerun_state_machine import RerunDataIterator
from runtime import ROOT, report

OUT=ROOT/'training-smoke/audit/multigpu-20261012'
events=[]
active=None
current_step=None
original_next=RerunDataIterator.__next__
original_step=training.train_step
original_eval=training.evaluate


def flush():
    if dist.is_initialized():
        (OUT/f"{os.environ['MULTIGPU_TAG']}-rank{dist.get_rank()}-events.json").write_text(json.dumps(events,indent=2)+'\n')


def capture_next(self):
    batch=original_next(self)
    if active is not None:
        h=hashlib.sha256()
        for key in ('tokens','labels','loss_mask','position_ids'):
            value=batch.get(key)
            if value is not None:
                h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        active['batch_hashes'].append(h.hexdigest())
    return batch


def capture_step(*args,**kwargs):
    global active,current_step
    params=training.get_args()
    current_step=kwargs['iteration']+1
    active={'kind':'train','iteration':current_step,'consumed_train_samples':params.consumed_train_samples,'batch_hashes':[]}
    try:
        result=original_step(*args,**kwargs)
        active.update(losses={k:float(v) for k,v in result[0].items()},skipped=result[1],grad_norm=float(result[5]) if result[5] is not None else None)
        active['parallel_ranks']={'tp':ps.get_tensor_model_parallel_rank(),'pp':ps.get_pipeline_model_parallel_rank(),'dp':ps.get_data_parallel_rank()}
        events.append(active);flush()
        return result
    finally:active=None


def capture_eval(*args,**kwargs):
    global active
    params=training.get_args()
    active={'kind':'eval','iteration':current_step if current_step is not None else params.iteration,'consumed_valid_before':params.consumed_valid_samples,'batch_hashes':[]}
    try:
        result=original_eval(*args,**kwargs)
        active.update(losses={k:float(v) for k,v in result[0].items()},consumed_valid_after=params.consumed_valid_samples)
        modules=unwrap_model(args[2] if len(args)>2 else kwargs['model'])
        if not isinstance(modules,list):modules=[modules]
        replica=hashlib.sha256();sample=hashlib.sha256();count=0
        for module in modules:
            for name,param in module.named_parameters():
                # Layer zero never calls the value-blend LoRA; the imported native
                # checkpoint has no values for its three unused placeholder parameters.
                if ps.is_pipeline_first_stage() and name.startswith('decoder.layers.0.self_attention.bottleneck_v.'):
                    continue
                flat=param.detach().reshape(-1)
                sample.update(name.encode());sample.update(flat[::max(1,flat.numel()//1024)][:1024].contiguous().view(torch.uint8).cpu().numpy().tobytes())
                if not getattr(param,'tensor_model_parallel',False):
                    replica.update(name.encode());replica.update(param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes());count+=1
        hashes=[None]*ps.get_tensor_model_parallel_world_size()
        dist.all_gather_object(hashes,replica.hexdigest(),group=ps.get_tensor_model_parallel_group())
        assert len(set(hashes))==1,('TP replicated parameters diverged',dist.get_rank(),hashes)
        dp_hashes=[None]*ps.get_data_parallel_world_size()
        dist.all_gather_object(dp_hashes,sample.hexdigest(),group=ps.get_data_parallel_group())
        assert len(set(dp_hashes))==1,('DP parameter sample diverged',dist.get_rank(),dp_hashes)
        active.update(tp_replicated_parameters_exact=True,replicated_parameters_checked=count,dp_parameter_sample_exact=True)
        active['parallel_ranks']={'tp':ps.get_tensor_model_parallel_rank(),'pp':ps.get_pipeline_model_parallel_rank(),'dp':ps.get_data_parallel_rank()}
        events.append(active);flush()
        return result
    finally:active=None


if __name__=='__main__':
    RerunDataIterator.__next__=capture_next
    training.train_step=capture_step
    training.evaluate=capture_eval
    runpy.run_path(str(ROOT/'pretrain_rwkv7.py'),run_name='__main__')
