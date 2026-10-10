"""Real TP loss-only test; no RWKV multi-GPU model or parallel carrier changes."""
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from rwkv_megatron.l2wrap import wrap_token_loss, max_logit_stats
from rwkv_megatron.rwkv7_model import RWKV7Model

PATH=Path(__file__).resolve().parent


def main():
    dist.init_process_group('gloo')
    rank=dist.get_rank();world=dist.get_world_size()
    assert world==2
    results=[]
    for scale in (1.,8.,0.):
        torch.manual_seed(19)
        full=torch.randn(5,2,12)
        # Winners on both shards, a cross-shard tie and all-negative logits.
        full[0,0,1]=full[0,0,8]=9
        full[1,0,9]=10
        full[2,0]-=full[2,0].max()+2
        labels=torch.randint(0,12,(5,2))
        mask=torch.ones(2,5);mask[0,3]=mask[1,1]=0
        shard=full[:,:,rank*6:(rank+1)*6].clone().requires_grad_()
        stats=max_logit_stats(shard,dist.group.WORLD)
        base=vocab_parallel_cross_entropy(shard,labels,tp_group=dist.group.WORLD).T.contiguous()
        out=wrap_token_loss(base,shard,1e-4,stats=stats)
        assert torch.equal(out,base)
        (scale*(out*mask).sum()/mask.sum()).backward()
        ref=full.clone().requires_grad_()
        ce=F.cross_entropy(ref.flatten(0,1),labels.flatten(),reduction='none').reshape(5,2).T
        penalty=.5e-4*ref.max(-1).values.T.square()
        (scale*((ce+penalty)*mask).sum()/mask.sum()).backward()
        expected=ref.grad[:,:,rank*6:(rank+1)*6]
        torch.testing.assert_close(shard.grad,expected,rtol=1e-5,atol=1e-7)
        # Exercise model interface using the real mutating vocab-parallel CE.
        model=RWKV7Model.__new__(RWKV7Model);torch.nn.Module.__init__(model)
        model.l2wrap_coefficient=1e-4;model.vocab_size=12;model.tp_group=dist.group.WORLD
        model.vocab_parallel_cross_entropy=lambda x,y,g:vocab_parallel_cross_entropy(x,y,tp_group=g)
        actual=full[:,:,rank*6:(rank+1)*6].clone().requires_grad_()
        losses=model.compute_language_model_loss(labels.T.contiguous(),actual)
        (scale*(losses*mask).sum()/mask.sum()).backward()
        torch.testing.assert_close(actual.grad,expected,rtol=1e-5,atol=1e-7)
        # Already gathered vocabulary: wrap locally, with no TP penalty collective.
        gathered=full.clone().requires_grad_()
        model.vocab_parallel_cross_entropy=lambda x,y,g:F.cross_entropy(x.flatten(0,1),y.flatten(),reduction='none').reshape(5,2)
        loss=model.compute_language_model_loss(labels.T.contiguous(),gathered)
        (scale*(loss*mask).sum()/mask.sum()).backward()
        torch.testing.assert_close(gathered.grad,ref.grad,rtol=1e-5,atol=1e-7)
        results.append({'scale':scale,'shard_max_abs_gradient_error':(actual.grad-expected).abs().max().item(),
                        'gathered_max_abs_gradient_error':(gathered.grad-ref.grad).abs().max().item()})
    if rank==0:
        (PATH/'l2wrap-tp-results.json').write_text(json.dumps({'world_size':world,'backend':'gloo CPU',
            'scope':'Loss only; real Megatron vocabulary-parallel CE plus L2Wrap; not full model parallel certification',
            'mask_and_cross_shard_tie_passed':True,'results':results},indent=2)+'\n')
        print('TP L2WRAP PASS',results,flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
