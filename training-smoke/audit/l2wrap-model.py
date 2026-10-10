"""Whole cropped RWKV model gradient vs independently differentiated penalty."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars

import distributed as audit
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model
from check import metric

PATH=Path(__file__).resolve().parent


def main():
    torch.cuda.set_device(0);dist.init_process_group('nccl')
    ps.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234);global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    audit.prepare_fixture();state=torch.load(PATH/'cropped-native.pth',weights_only=True)
    torch.manual_seed(19);tokens=torch.randint(1,128,(2,129),device='cuda')
    x,y=tokens[:,:-1],tokens[:,1:];mask=torch.ones_like(y,dtype=torch.float32);mask[:,::7]=0
    rows={}
    for bf16,recompute in ((False,False),(True,False),(True,True)):
        cfg=audit.config_for(1,1,False,recompute,fp32=not bf16)
        cfg.rwkv_l2wrap_coefficient=1e-4;cfg.rwkv_key_fusion=True
        model=RWKV7Model(cfg,transformer_layer_spec=get_rwkv7_layer_spec(),vocab_size=128,
            max_sequence_length=128,pre_process=True,post_process=True,parallel_output=False,
            share_embeddings_and_output_weights=False,position_embedding_type='none').cuda().to(cfg.params_dtype)
        model.from_rwkv_lm(state,False);model.train()
        position=torch.arange(128,device=x.device).expand_as(x)
        actual=model(x,position,None,labels=y)
        (3*(actual*mask).sum()/mask.sum()).backward()
        actual_grad={n:p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n,p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        logits=model(x,position,None).transpose(0,1).contiguous()
        # Keep original logits for the independent penalty; clone shields them
        # from the unchanged native FP32 CE's in-place work buffer.
        ce=super(RWKV7Model,model).compute_language_model_loss(y,logits.clone())
        penalty=.5e-4*logits.float().max(-1).values.T.square()
        assert torch.equal(actual,ce)
        (3*((ce+penalty)*mask).sum()/mask.sum()).backward()
        metrics={n:metric(actual_grad[n],p.grad if p.grad is not None else torch.zeros_like(p)) for n,p in model.named_parameters()}
        agg=(sum((m['relative_l2']*m['reference_l2'])**2 for m in metrics.values())/sum(m['reference_l2']**2 for m in metrics.values()))**.5
        assert all(m['finite'] for m in metrics.values())
        assert max(m['relative_l2'] for m in metrics.values())<1e-4
        name=f'{"bf16" if bf16 else "fp32"}-recompute{recompute}'
        rows[name]={'coefficient':1e-4,'forward_ce_exact':True,'parameters':len(metrics),
                    'aggregate_gradient_relative_l2':agg,'gradient_by_parameter':metrics}
        print('WHOLE MODEL L2WRAP',name,'gradient error',agg,flush=True)
        del model;torch.cuda.empty_cache()
    (PATH/'l2wrap-model-results.json').write_text(json.dumps({'shape':'4 layers hidden256 batch2 seq128',
        'oracle':'Same CE backend + independent differentiable 0.5e-4*max(logit)^2 per token',
        'mask':'Every 7th token excluded','upstream_scale':3,'results':rows},indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
