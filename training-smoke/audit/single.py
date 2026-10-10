"""Compare a single-GPU Megatron RWKV7 with the native torch reference."""
import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars
import rwkv_megatron.rwkv7_layer_specs as specs
from rwkv_megatron.rwkv7_model import RWKV7Model
from rwkv_conversion import convert_pipeline_stage
import distributed as audit
from native_reference import NativeReference
from check import metric
from runtime import report

PATH=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true')
    parser.add_argument('--all-gradients',action='store_true')
    parser.add_argument('--bf16',action='store_true');parser.add_argument('--naive-core',action='store_true')
    parser.add_argument('--torch-norms',action='store_true')
    parser.add_argument('--aligned-reference',action='store_true')
    parser.add_argument('--apex-reference-norms',action='store_true')
    parser.add_argument('--te-reference-norms',action='store_true')
    parser.add_argument('--result-tag',default='')
    parser.add_argument('--capture-output',help='Optional audit capture basename (stored under the audit directory)')
    parser.add_argument('--capture-core-gradients',action='store_true')
    parser.add_argument('--audit-v0-edges',action='store_true')
    parser.add_argument('--split-value-reference',action='store_true')
    parser.add_argument('--shared-value-reference',action='store_true',help='Historical negative control: share WKV/correction value view')
    parser.add_argument('--reference-profile',choices=('blinkdl','diagnostic'),help='Default: BlinkDL source bodies; diagnostic flags retain their explicit shared-primitive mode')
    parser.add_argument('--shared-pointops',action='store_true')
    parser.add_argument('--te-final-reference',action='store_true')
    parser.add_argument('--carrier-chain-reference',action='store_true')
    parser.add_argument('--reference-core',choices=('recurrent','fla'),default='recurrent')
    args=parser.parse_args()
    diagnostic_flags=any((args.aligned_reference,args.apex_reference_norms,args.te_reference_norms,
                          args.shared_pointops,args.te_final_reference,args.carrier_chain_reference,
                          args.split_value_reference,args.shared_value_reference))
    profile=args.reference_profile or ('diagnostic' if diagnostic_flags else 'blinkdl')
    if profile=='blinkdl' and diagnostic_flags:
        parser.error('BlinkDL reference cannot use Megatron-aligned norm/pointop/carrier flags')
    if args.split_value_reference and args.shared_value_reference:
        parser.error('split-value-reference and shared-value-reference are mutually exclusive')
    torch.cuda.set_device(0);dist.init_process_group('nccl')
    ps.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234);torch.backends.cuda.matmul.allow_tf32=False
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    audit.FULL=args.full
    if args.full:audit.DIMS={'layers':32,'hidden':2560,'heads':40,'ffn':10240,'vocab':65536,'sequence':128,'batch':1,'microbatches':1}
    dim=audit.DIMS;dtype=torch.bfloat16 if args.bf16 else torch.float32
    if args.full:
        state=torch.load('/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth',map_location='cpu',weights_only=True)
        tokens=torch.load(PATH/'minipile-tokens.pth',weights_only=True).cuda()
    else:
        audit.prepare_fixture();state=torch.load(PATH/'cropped-native.pth',weights_only=True)
        gen=torch.Generator().manual_seed(20261009);tokens=torch.randint(1,128,(2,129),generator=gen).cuda()
    x,y=tokens[:,:-1],tokens[:,1:]
    cfg=audit.config_for(1,1,False,False,fp32=not args.bf16)
    model=RWKV7Model(cfg,transformer_layer_spec=specs.get_rwkv7_layer_spec(),vocab_size=dim['vocab'],
        max_sequence_length=128,pre_process=True,post_process=True,parallel_output=False,
        share_embeddings_and_output_weights=False,position_embedding_type='none').cuda().to(dtype)
    model.from_rwkv_lm(state,False);model.train()
    if args.torch_norms:
        norm_modules=[model.pre_norm,model.decoder.final_layernorm]
        for layer in model.decoder.layers:norm_modules.extend([layer.input_layernorm,layer.pre_mlp_layernorm])
        for module in norm_modules:
            def norm_forward(x,module=module):
                return F.layer_norm(x,(x.shape[-1],),module.weight,module.bias,1e-5)
            module.forward=norm_forward
    trace={};handles=[];retained={}
    def hook(key,transform=lambda t:t.transpose(0,1)):
        def capture(module,inputs,output):
            tensor=output[0] if isinstance(output,tuple) else output
            trace[key]=transform(tensor).detach().cpu()
            if tensor.requires_grad:
                tensor.retain_grad();retained[key]=(tensor,transform)
        return capture
    handles.append(model.pre_norm.register_forward_hook(hook('ln0')))
    handles.append(model.decoder.final_layernorm.register_forward_hook(hook('final-norm',lambda t:t[0].transpose(0,1) if t.ndim==4 else t.transpose(0,1))))
    for layer in model.decoder.layers:
        index=layer.self_attention.layer_number-1
        if args.naive_core:layer.self_attention.core_attention.forward=audit.recurrence
        for key,module in [('norm1',layer.input_layernorm),('norm2',layer.pre_mlp_layernorm),
                           ('r',layer.self_attention.linear_receptance),('k-raw',layer.self_attention.linear_key),
                           ('v-raw',layer.self_attention.linear_value),('wkv',layer.self_attention.core_attention),
                           ('mlp',layer.mlp)]:
            handles.append(module.register_forward_hook(hook(f'{index}.{key}')))
        def group_post(module,inputs,output,index=index):
            trace[f'{index}.groupnorm']=output.reshape(x.shape[1],x.shape[0],-1).transpose(0,1).detach().cpu()
        handles.append(layer.self_attention.group_norm.register_forward_hook(group_post))
        def project_pre(module,inputs,index=index):
            trace[f'{index}.project-input']=inputs[0].transpose(0,1).detach().cpu()
        handles.append(layer.self_attention.linear_proj.register_forward_pre_hook(project_pre))
        def core_pre(module,inputs,index=index):
            for name,tensor in zip(('r','k','v','decay','a','b'),inputs):
                trace[f'{index}.core-{name}']=tensor.transpose(0,1).detach().cpu()
                if args.capture_core_gradients and tensor.requires_grad:
                    tensor.retain_grad()
                    retained[f'{index}.core-{name}']=(tensor,lambda t:t.transpose(0,1))
        handles.append(layer.self_attention.core_attention.register_forward_pre_hook(core_pre))
        handles.append(layer.self_attention.register_forward_hook(hook(f'{index}.att',lambda t:t[0].transpose(0,1))))
        handles.append(layer.register_forward_hook(hook(f'{index}.hidden',lambda t:t[0].transpose(0,1))))
    logits=model(x,torch.arange(x.shape[1],device=x.device).expand_as(x),None)
    loss=F.cross_entropy(logits.float().reshape(-1,dim['vocab']),y.reshape(-1))
    if args.audit_v0_edges:
        from v0_edges import V0Edges, summarize
        edge_audit=V0Edges(loss,retained['0.v-raw'][0])
    loss.backward()
    if args.audit_v0_edges:actual_edges=edge_audit.finish()
    selected=None
    if args.full and not args.all_gradients:
        selected={'blocks.0.att.value.weight','blocks.0.ln0.weight','blocks.0.ln0.bias','ln_out.weight','ln_out.bias'}
        for layer in range(dim['layers']):
            selected.update(f'blocks.{layer}.{name}' for name in ('ln1.weight','ln1.bias','ln2.weight','ln2.bias'))
            selected.update(f'blocks.{layer}.att.x_{ch}' for ch in 'rwkvag')
            if layer:selected.add(f'blocks.{layer}.att.v0')
    source={key:(p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p).cpu())
            for key,p in model.named_parameters()}
    actual_grad=convert_pipeline_stage([source],0,cfg,0)
    if selected:actual_grad={k:v for k,v in actual_grad.items() if k in selected}
    del source
    actual_logits=logits.detach().cpu();actual_loss=loss.item()
    first_value_grad=retained['0.v-raw'][0].grad.transpose(0,1).detach().cpu()
    actual_grad_trace={key:transform(tensor.grad).detach().cpu() for key,(tensor,transform) in retained.items() if tensor.grad is not None}
    if args.capture_output:
        if Path(args.capture_output).name != args.capture_output:
            raise ValueError('capture-output must be a basename')
        torch.save(dict(logits=actual_logits,loss=actual_loss,gradients=actual_grad,
                        trace=trace,v0_gradient=first_value_grad,gradient_trace=actual_grad_trace),
                   PATH/(args.capture_output+'.pth'))
    retained.clear()
    for handle in handles:handle.remove()
    del logits,loss,model;gc.collect();torch.cuda.empty_cache()
    if profile=='blinkdl':
        from blinkdl_reference import BlinkDLReference
        reference=BlinkDLReference(state,dtype)
    else:
        reference=NativeReference(state,dtype,core=args.reference_core,aligned=args.aligned_reference,apex_norms=args.apex_reference_norms,shared_pointops=args.shared_pointops,te_final=args.te_final_reference,carrier_chain=args.carrier_chain_reference,te_norms=args.te_reference_norms,split_value_views=not args.shared_value_reference)
    ref_logits=reference(x);ref_loss=F.cross_entropy(ref_logits.float().reshape(-1,dim['vocab']),y.reshape(-1))
    if args.audit_v0_edges:edge_audit=V0Edges(ref_loss,reference.v_first)
    ref_loss.backward();ref_grad=reference.native_gradients(selected)
    if args.audit_v0_edges:reference_edges=edge_audit.finish()
    trace_metrics={k:metric(v,reference.trace[k]) for k,v in trace.items() if k in reference.trace}
    gradient_trace={k:metric(v,(reference.live[k].grad.transpose(0,1) if args.aligned_reference else reference.live[k].grad).detach().cpu()) for k,v in actual_grad_trace.items() if k in reference.live and reference.live[k].grad is not None}
    grad_metrics={k:metric(actual_grad[k],v) for k,v in ref_grad.items()}
    result={'full_3b':args.full,'dtype':str(dtype),'megatron_core':'torch recurrence' if args.naive_core else 'FLA',
        'reference_core':('official CUDA' if args.bf16 else 'independent torch recurrence') if profile=='blinkdl' else args.reference_core,
        'reference_profile':profile,'reference_provenance':reference.provenance() if profile=='blinkdl' else {'kind':'shared-primitive diagnostic; not an official implementation'},
        'torch_norms':args.torch_norms,'runtime':report(),'loss_megatron':actual_loss,
        'loss_native':ref_loss.item(),'loss_delta':actual_loss-ref_loss.item(),
        'gradient_trace':gradient_trace,'logits':metric(actual_logits,ref_logits.detach().cpu()),'trace':trace_metrics,
        'gradient_by_parameter':grad_metrics,'gradient_scope':'selected' if selected else 'all',
        'v0_reference_gradient_norm':reference.v_first.grad.float().norm().item(),
        'first_value_gradient':metric(first_value_grad,reference.v_first.grad.transpose(0,1).detach().cpu() if args.aligned_reference else reference.v_first.grad.detach().cpu()),
        'max_relative_gradient':max(v['relative_l2'] for v in grad_metrics.values())}
    reference_name=('blinkdl-main' if args.bf16 else 'blinkdl-simplified') if profile=='blinkdl' else args.reference_core
    name=('full-' if args.full else 'small-')+('bf16-' if args.bf16 else 'fp32-')+('naive' if args.naive_core else 'fla')+'-vs-'+reference_name
    name+=('-all-gradients' if args.all_gradients else '')
    name+=('-torch-norms' if args.torch_norms else '')+('-aligned' if args.aligned_reference else '')+('-apex' if args.apex_reference_norms else '')+('-shared-pointops' if args.shared_pointops else '')+('-te-final' if args.te_final_reference else '')+('-carrier-chain' if args.carrier_chain_reference else '')
    name+=('-te-norms' if args.te_reference_norms else '')+args.result_tag
    name+=('-split-value' if args.split_value_reference else '')
    name+=('-shared-value' if args.shared_value_reference else '')
    name+=('-blinkdl-source' if profile=='blinkdl' else '')
    if args.audit_v0_edges:
        result['v0_edges']={'actual':summarize(actual_edges),'reference':summarize(reference_edges)}
        torch.save({'actual':actual_edges,'reference':reference_edges},PATH/(name+'-v0-edges.pth'))
    (PATH/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('SINGLE RESULT',name,'losses',actual_loss,ref_loss.item(),'logits',result['logits'],flush=True)
    print('FIRST TRACES',list(trace_metrics.items())[:5],flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
