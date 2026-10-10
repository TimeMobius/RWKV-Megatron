"""Compare the new loss interface to pinned BlinkDL's actual CUDA loss."""
import ast
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from blinkdl_reference import SOURCE, COMMIT
from check import metric
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from rwkv_megatron.rwkv7_model import RWKV7Model

PATH=Path(__file__).resolve().parent


def official_loss():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    def isolated_load(**kwargs):
        kwargs['sources']=[str(SOURCE/name) for name in kwargs['sources']]
        build=PATH.parent/'cache/torch-extensions-blinkdl'/kwargs['name']
        build.mkdir(parents=True,exist_ok=True);kwargs['build_directory']=str(build)
        kwargs['extra_include_paths']=[str(Path(os.environ['CUDA_HOME'])/'targets/x86_64-linux/include')]
        kwargs['extra_include_paths'].extend(str(p) for p in sorted((Path(torch.__file__).parent.parent/'nvidia').glob('*/include')))
        os.environ.setdefault('MAX_JOBS','2')
        return load(**kwargs)
    source=SOURCE/'src/model.py';tree=ast.parse(source.read_text())
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='L2WRAP_CE_CUDA_V2' for t in node.targets):selected.append(node)
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ('L2WrapCrossEntropyCUDA','l2wrap_cross_entropy'):selected.append(node)
    assert len(selected)==3
    namespace={'torch':torch,'load':isolated_load}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['l2wrap_cross_entropy']


def main():
    torch.cuda.set_device(0);dist.init_process_group('nccl')
    loss_fn=official_loss()
    capture=torch.load(PATH/'full-production-key-fusion.pth',map_location='cpu',weights_only=True,mmap=True)
    tokens=torch.load(PATH/'minipile-tokens.pth',weights_only=True).cuda()
    labels=tokens[:,1:]
    model=RWKV7Model.__new__(RWKV7Model);torch.nn.Module.__init__(model)
    model.l2wrap_coefficient=1e-4;model.vocab_size=65536;model.tp_group=dist.group.WORLD
    model.vocab_parallel_cross_entropy=lambda x,y,g:vocab_parallel_cross_entropy(x,y,tp_group=g)
    rows={}
    for dtype in (torch.float32,torch.bfloat16):
        for scale in (1.,8.):
            x=capture['logits'].cuda().to(dtype).transpose(0,1).contiguous().requires_grad_()
            ref=capture['logits'].cuda().to(dtype).requires_grad_()
            actual=model.compute_language_model_loss(labels,x).mean()
            official=loss_fn(ref,labels)
            (actual*scale).backward();(official*scale).backward()
            gradient=metric(x.grad.transpose(0,1),ref.grad)
            row={'loss':actual.item(),'official_loss':official.item(),'loss_abs_delta':abs(actual.item()-official.item()),
                 'gradient':gradient,'scale':scale}
            # BF16 separate CE/extra-gradient rounding can differ from the fused
            # official CUDA write. FP32 checks the formula without that cast.
            threshold=1e-5 if dtype==torch.float32 else .004
            assert gradient['finite'] and gradient['relative_l2']<threshold,row
            assert row['loss_abs_delta']<1e-5,row
            rows[f'{dtype}-scale{scale}']=row
            print('OFFICIAL L2WRAP',dtype,scale,row,flush=True)
    (PATH/'l2wrap-official-results.json').write_text(json.dumps({'repository':'BlinkDL/RWKV-LM','commit':COMMIT,
        'source':'RWKV-v7/train_temp/src/model.py::L2WrapCrossEntropyCUDA',
        'input':'Saved full 3B seq128 logits, matching MiniPile labels; same input per loss backend',
        'coefficient':1e-4,'results':rows},indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
