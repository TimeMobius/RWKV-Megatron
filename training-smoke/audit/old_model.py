"""Freeze before-update full-model outputs/gradients on the shared audit fixture."""
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
import fla
import megatron.core
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import global_vars
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'training-smoke/audit'


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    ps.initialize_model_parallel(tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    global_vars._GLOBAL_ARGS=SimpleNamespace(transformer_impl='local')
    origins=dict(python=sys.executable,megatron=str(Path(megatron.core.__file__).resolve()),
                 fla=str(Path(fla.__file__).resolve()),model=str(Path(sys.modules[RWKV7Model.__module__].__file__).resolve()))
    (OUT/'baseline-runtime.json').write_text(json.dumps(origins,indent=2)+'\n')
    cfg=TransformerConfig(num_layers=4,hidden_size=256,num_attention_heads=4,ffn_hidden_size=1024,
        kv_channels=64,tensor_model_parallel_size=1,pipeline_model_parallel_size=1,
        sequence_parallel=False,bf16=True,params_dtype=torch.bfloat16,pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.,attention_dropout=0.,add_bias_linear=False,add_qkv_bias=False,
        perform_initialization=False,gradient_accumulation_fusion=False,
        bias_dropout_fusion=True,persist_layer_norm=False,layernorm_epsilon=1e-5,
        deallocate_pipeline_outputs=False,enable_autocast=False)
    for name in ('D_AAA_LORA','D_DECAY_LORA','D_GATE_LORA','D_MV_LORA'):setattr(cfg,name,16)
    model=RWKV7Model(cfg,transformer_layer_spec=get_rwkv7_layer_spec(),vocab_size=128,
                    max_sequence_length=128,pre_process=True,post_process=True,
                    parallel_output=False,share_embeddings_and_output_weights=False,
                    position_embedding_type='none').cuda().to(torch.bfloat16)
    model.from_rwkv_lm(torch.load(OUT/'cropped-native.pth',weights_only=True),False)
    model.train()
    gen=torch.Generator(device='cpu').manual_seed(20261009)
    tokens=[torch.randint(1,128,(2,128),generator=gen).cuda() for _ in range(2)]
    labels=[torch.randint(1,128,(2,128),generator=gen).cuda() for _ in range(2)]
    outputs=[];losses=[]
    for token,label in zip(tokens,labels):
        logits=model(token,torch.arange(128,device=token.device).expand_as(token),None)
        loss=F.cross_entropy(logits.float().reshape(-1,128),label.reshape(-1))
        outputs.append(logits.detach().cpu());losses.append(loss.detach().cpu())
        (loss/2).backward()
    spec=importlib.util.spec_from_file_location('audit_converter',ROOT/'rwkv_conversion.py')
    converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    source={key:(param.grad.detach().cpu() if param.grad is not None else torch.zeros_like(param).cpu())
            for key,param in model.named_parameters()}
    gradients=converter.convert_pipeline_stage([source],0,cfg,0)
    torch.save(dict(name='baseline',tp=1,pp=1,sp=False,recompute=False,native_grads=gradients,
                    logits=torch.stack(outputs),losses=torch.stack(losses),ranks=[]),OUT/'baseline.pth')
    print('BASELINE AUDIT COMPLETE',losses,flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
