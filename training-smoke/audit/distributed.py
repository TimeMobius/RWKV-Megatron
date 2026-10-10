"""Real RWKV forward/backward, SP, PP and v0 numerical audit on identical weights."""
import argparse
import gc
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from runtime import ROOT, report
from megatron.core import parallel_state as ps
from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import global_vars
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model
from rwkv_conversion import convert_pipeline_stage

OUT = ROOT/'training-smoke/audit'
FULL=False
MINIPILE=False
DIMS={'layers':4,'hidden':256,'heads':4,'ffn':1024,'vocab':128,'sequence':128,'batch':2,'microbatches':2}


def metric(actual, expected):
    actual, expected = actual.float(), expected.float()
    diff = actual - expected
    return dict(max_abs=diff.abs().max().item(),
                relative_l2=(diff.norm() / expected.norm().clamp_min(1e-12)).item(),
                reference_l2=expected.norm().item(), actual_l2=actual.norm().item(),
                finite=bool(torch.isfinite(actual).all()))


def recurrence(r, k, v, decay, a, b):
    # Full FP32 state recurrence with ordinary torch autograd, no FLA kernels.
    dtype = r.dtype
    r, k, v, decay, a, b = [x.float().transpose(0,1).reshape(x.shape[1],x.shape[0],-1,64)
                            for x in (r,k,v,decay,a,b)]
    state = torch.zeros(r.shape[0],r.shape[2],64,64,device=r.device)
    result=[]
    for t in range(r.shape[1]):
        correction=(state*a[:,t].unsqueeze(-1)).sum(-2)
        state = state * (-decay[:,t]).exp().unsqueeze(-1)
        state = state + k[:,t].unsqueeze(-1)*v[:,t].unsqueeze(-2)
        state = state + b[:,t].unsqueeze(-1)*correction.unsqueeze(-2)
        result.append((state*r[:,t].unsqueeze(-1)).sum(-2))
    return torch.stack(result,dim=1).flatten(2).transpose(0,1).contiguous().to(dtype)


def prepare_fixture():
    fixture=OUT/'cropped-native.pth'
    if fixture.exists(): return
    src=torch.load('/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth',
                   map_location='cpu',weights_only=True)
    native={}
    for name,value in src.items():
        if name.startswith('blocks.') and int(name.split('.')[1])>=4: continue
        if name in ('emb.weight','head.weight'): value=value[:128,:256]
        elif name.endswith('att.r_k'): value=value[:4,:64]
        elif name.endswith(('.ffn.key.weight',)): value=value[:1024,:256]
        elif name.endswith(('.ffn.value.weight',)): value=value[:256,:1024]
        elif name.endswith('.weight') and value.ndim==2: value=value[:256,:256]
        elif name.endswith(('att.w1','att.a1','att.g1','att.v1')): value=value[:256,:16]
        elif name.endswith(('att.w2','att.a2','att.g2','att.v2')): value=value[:16,:256]
        elif value.ndim==3: value=value[:,:,:256]
        elif value.ndim==1: value=value[:256]
        native[name]=value.clone().to(torch.bfloat16)
    torch.save(native,fixture)


def prepare_minipile():
    import numpy as np
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    cache=ROOT/'training-smoke/cache'
    prefix=cache/'e9841bdde4d0eda1dac255c6df373f3b-GPTDataset-train'
    doc=np.load(str(prefix)+'-document_index.npy',mmap_mode='r')
    sample=np.load(str(prefix)+'-sample_index.npy',mmap_mode='r')
    shuffle=np.load(str(prefix)+'-shuffle_index.npy',mmap_mode='r')
    index,offset=sample[int(shuffle[0])]
    data=IndexedDataset('/mnt/data/Datasets/minipile/minipile_text_document')
    chunks=[];remaining=DIMS['sequence']+1
    while remaining:
        document=int(doc[index]);length=min(remaining,int(data.sequence_lengths[document])-int(offset))
        chunks.append(data.get(document,offset=int(offset),length=length).astype(np.int64))
        remaining-=length;index+=1;offset=0
    tokens=torch.from_numpy(np.concatenate(chunks)).unsqueeze(0)
    torch.save(tokens,OUT/'minipile-tokens.pth')
    (OUT/'minipile-input.json').write_text(json.dumps({'cache_prefix':str(prefix),'sample_index':int(shuffle[0]),
        'sequence_length':DIMS['sequence'],'token_ids':tokens[0].tolist()},indent=2)+'\n')


def config_for(tp, pp, sp, recompute, fp32=False):
    dtype=torch.float32 if fp32 else torch.bfloat16
    config=TransformerConfig(num_layers=DIMS['layers'],hidden_size=DIMS['hidden'],num_attention_heads=DIMS['heads'],ffn_hidden_size=DIMS['ffn'],
        kv_channels=64,tensor_model_parallel_size=tp,pipeline_model_parallel_size=pp,
        sequence_parallel=sp,bf16=not fp32,params_dtype=dtype,pipeline_dtype=dtype,
        hidden_dropout=0.,attention_dropout=0.,add_bias_linear=False,add_qkv_bias=False,
        perform_initialization=False,gradient_accumulation_fusion=False,
        bias_dropout_fusion=True,persist_layer_norm=False,layernorm_epsilon=1e-5,
        deallocate_pipeline_outputs=False,enable_autocast=False,
        recompute_granularity='full' if recompute else None,
        recompute_method='uniform' if recompute else None,
        recompute_num_layers=1 if recompute else None)
    for name,size in [('D_AAA_LORA',96),('D_DECAY_LORA',96),('D_GATE_LORA',320),('D_MV_LORA',64)]:setattr(config,name,size if FULL else 16)
    return config


def run_case(name,tp,pp,sp=False,recompute=False,detach_v0=False,naive=False):
    ps.initialize_model_parallel(tensor_model_parallel_size=tp,pipeline_model_parallel_size=pp)
    model_parallel_cuda_manual_seed(1234)
    cfg=config_for(tp,pp,sp,recompute,fp32="fp32" in name)
    native=torch.load('/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth' if FULL else OUT/'cropped-native.pth',map_location='cpu',weights_only=True)
    model=RWKV7Model(cfg,transformer_layer_spec=get_rwkv7_layer_spec(),vocab_size=DIMS['vocab'],
                    max_sequence_length=DIMS['sequence'],pre_process=ps.is_pipeline_first_stage(),
                    post_process=ps.is_pipeline_last_stage(),parallel_output=False,
                    share_embeddings_and_output_weights=False,position_embedding_type='none')
    model=model.cuda().to(cfg.params_dtype)
    model.from_rwkv_lm(native,False)
    model.train()
    model.ddp_config=SimpleNamespace(use_megatron_fsdp=False)
    records={}
    handles=[]
    def retain(tag,tensor):
        if torch.is_grad_enabled() and tensor.requires_grad:
            tensor.retain_grad()
            records.setdefault(tag,[]).append(tensor)
    for layer in model.decoder.layers:
        index=layer.self_attention.layer_number-1
        if naive:layer.self_attention.core_attention.forward=recurrence
        if FULL and index not in (0,1,15,16,31):continue
        def pre_hook(module,args,index=index):
            if args[0].ndim==4:retain(f'input-{index}',args[0])
        def post_hook(module,args,output,index=index):
            carrier=output[0]
            if detach_v0 and index==0:
                carrier=torch.cat((carrier[:1],carrier[1:].detach()),dim=0)
                output=(carrier,output[1])
            retain(f'output-{index}',carrier)
            return output
        handles.append(layer.self_attention.register_forward_pre_hook(pre_hook))
        handles.append(layer.self_attention.register_forward_hook(post_hook))
        if index==0:
            handles.append(layer.self_attention.linear_value.register_forward_hook(
                lambda m,a,o:retain('first-value',o[0])))
    if not model.pre_process:
        def pp_pre(module,args):retain('pp-recv',module.input_tensor)
        handles.append(model.decoder.register_forward_pre_hook(pp_pre))
    if not model.post_process:
        handles.append(model.register_forward_hook(lambda m,a,o:retain('pp-send',o)))
    gen=torch.Generator(device='cpu').manual_seed(20261009)
    tokens=[torch.randint(1,DIMS['vocab'],(DIMS['batch'],DIMS['sequence']),generator=gen).cuda() for _ in range(DIMS['microbatches'])]
    labels=[torch.randint(1,DIMS['vocab'],(DIMS['batch'],DIMS['sequence']),generator=gen).cuda() for _ in range(DIMS['microbatches'])]
    if MINIPILE:
        sample=torch.load(OUT/'minipile-tokens.pth',weights_only=True).cuda()
        tokens=[sample[:,:-1]];labels=[sample[:,1:]]
    outputs=[];losses=[]
    def forward(iterator,module):
        token,label=next(iterator)
        logits=module(token,torch.arange(DIMS['sequence'],device=token.device).expand_as(token),None)
        if module.post_process:outputs.append(logits.detach().cpu())
        def loss_fn(out):
            loss=F.cross_entropy(out.float().reshape(-1,DIMS['vocab']),label.reshape(-1))
            losses.append(loss.detach().cpu())
            return loss, {'audit-loss':loss.detach()}
        return logits,loss_fn
    get_forward_backward_func()(forward_step_func=forward,data_iterator=iter(zip(tokens,labels)),
        model=model,num_microbatches=DIMS['microbatches'],seq_length=DIMS['sequence'],micro_batch_size=DIMS['batch'],forward_only=False)
    _allreduce_non_tensor_model_parallel_grads([model],cfg,ps.get_tensor_model_parallel_group())
    grads={key:(param.grad.detach().cpu() if param.grad is not None else torch.zeros_like(param).cpu())
           for key,param in model.named_parameters() if select_gradient(key,model)}
    captures={key:[dict(value=t.detach().cpu(),grad=t.grad.detach().cpu() if t.grad is not None else None)
                   for t in values] for key,values in records.items()}
    payload=dict(name=name,tp=ps.get_tensor_model_parallel_rank(),pp=ps.get_pipeline_model_parallel_rank(),
                 dp=ps.get_data_parallel_rank(),grads=grads,outputs=outputs,losses=losses,captures=captures)
    gathered=[None]*dist.get_world_size() if dist.get_rank()==0 else None
    dist.gather_object(payload,gathered,dst=0)
    if dist.get_rank()==0:
        selected=[x for x in gathered if x['dp']==0]
        config=SimpleNamespace(num_layers=DIMS['layers'],hidden_size=DIMS['hidden'],num_attention_heads=DIMS['heads'],
                               tensor_model_parallel_size=tp,pipeline_model_parallel_size=pp)
        merged={}
        for stage in range(pp):
            shards=sorted([x for x in selected if x['pp']==stage],key=lambda x:x['tp'])
            merged.update(convert_pipeline_stage([x['grads'] for x in shards],stage,config,0))
        final=next(x for x in selected if x['pp']==pp-1 and x['tp']==0)
        result=dict(name=name,tp=tp,pp=pp,sp=sp,recompute=recompute,detach_v0=detach_v0,full_3b=FULL,dimensions=DIMS.copy(),minipile=MINIPILE,
                    naive=naive,native_grads=merged,logits=torch.stack(final['outputs']),
                    losses=torch.stack(final['losses']),ranks=selected)
        torch.save(result,OUT/f'{"full-3b-minipile-" if MINIPILE else "full-3b-" if FULL else ""}{name}.pth')
        print(f'AUDIT COMPLETED {name}: losses={final["losses"]}',flush=True)
    for handle in handles:handle.remove()
    dist.barrier()
    del model,native,grads,records,captures,payload,gathered
    gc.collect();torch.cuda.empty_cache()
    ps.destroy_model_parallel()
    dist.barrier()


def select_gradient(key,model):
    if not FULL:return True
    if not key.startswith('decoder.layers.'):
        return 'norm.' in key or 'layernorm.' in key
    index=int(key.split('.')[2])+model.decoder.layers[0].self_attention.layer_number-1
    return ('layernorm.' in key or key.endswith(('mix_coeff_input','bottleneck_v.bias'))
            or (index==0 and key.endswith('linear_value.weight')))


def main():
    global FULL,DIMS,MINIPILE
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['dense','dense-detached','dense-naive','tp2','tp2-sp',
        'pp2','tp2-pp2','tp2-sp-pp2','tp2-sp-pp2-recompute','tp4-sp-pp2'])
    parser.add_argument('--full-3b',action='store_true')
    parser.add_argument('--minipile',action='store_true')
    args=parser.parse_args()
    FULL=args.full_3b
    MINIPILE=args.minipile
    assert not MINIPILE or FULL
    if FULL:DIMS={'layers':32,'hidden':2560,'heads':40,'ffn':10240,'vocab':65536,'sequence':128,'batch':1,'microbatches':1}
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
    torch.backends.cuda.matmul.allow_tf32=False
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    runtime=report()
    all_runtime=[None]*dist.get_world_size() if dist.get_rank()==0 else None
    dist.gather_object(runtime,all_runtime,dst=0)
    if dist.get_rank()==0:
        (OUT/'distributed-runtime.json').write_text(json.dumps(all_runtime,indent=2)+'\n')
        prepare_fixture()
        if MINIPILE:prepare_minipile()
    dist.barrier()
    cases={'dense':(1,1,False,False,False,False),'dense-detached':(1,1,False,False,True,False),
           'dense-naive':(1,1,False,False,False,True),'tp2':(2,1,False,False,False,False),
           'tp2-sp':(2,1,True,False,False,False),'pp2':(1,2,False,False,False,False),
           'tp2-pp2':(2,2,False,False,False,False),'tp2-sp-pp2':(2,2,True,False,False,False),
           'tp2-sp-pp2-recompute':(2,2,True,True,False,False),
           'tp4-sp-pp2':(4,2,True,False,False,False),
           'dense-fp32-naive':(1,1,False,False,False,True),
           'dense-fp32-detached':(1,1,False,False,True,True),
           'tp2-fp32-naive':(2,1,False,False,False,True),
           'tp2-sp-pp2-fp32-naive':(2,2,True,False,False,True),
           'tp2-sp-pp2-recompute-fp32-naive':(2,2,True,True,False,True),
           'tp4-sp-pp2-fp32-naive':(4,2,True,False,False,True)}
    for name in args.cases:run_case(name,*cases[name])
    dist.destroy_process_group()


if __name__=='__main__':main()
