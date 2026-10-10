"""Read-only source history and runtime eligibility of optional FLA features."""
import importlib.util
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.modules.l2norm import l2_norm
from fla.ops.generalized_delta_rule.dplr.backends.tilelang import DPLRTileLangBackend

from check import metric

PATH=Path(__file__).resolve().parent
REPO=PATH.parents[1]/'3rdparty/flash-linear-attention-260115'


def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()


def main():
    torch.cuda.set_device(0)
    features={}
    for name,path in {
        'output_correction':'fla/ops/rwkv7/gate_output_correction.py',
        'channel_mixing':'fla/ops/rwkv7/channel_mixing.py',
        'l2_norm':'fla/modules/l2norm.py',
        'fused_linear_ce':'fla/modules/fused_linear_cross_entropy.py',
        'tilelang_backend':'fla/ops/generalized_delta_rule/dplr/backends/tilelang/__init__.py',
    }.items():
        exists=subprocess.run(['git','-C',str(REPO),'cat-file','-e',f'muon:{path}'],capture_output=True).returncode==0
        features[name]={'path':path,'exists_in_original_muon':exists,
                        'history_since_muon':git('log','--oneline','muon..HEAD','--',path).splitlines()}
    backend=DPLRTileLangBackend();eligibility=[]
    for batch in (1,2):
        values=[torch.zeros(batch,128,40,64,device='cuda',dtype=torch.bfloat16) for _ in range(6)]
        for chunk_size in (16,32,64):
            for bound in (None,-0.6065306597126334):
                accepted,reason=backend.chunk_dplr_delta_rule_verifier(*values,safe_gate=True,
                    chunk_size=chunk_size,lower_bound=bound)
                eligibility.append({'batch':batch,'heads':40,'head_size':64,'chunk_size':chunk_size,
                    'lower_bound':bound,'shape_eligible':accepted,'reason':reason})
    capture=torch.load(PATH/'full-production-key-fusion.pth',map_location='cpu',weights_only=True,mmap=True)
    state=torch.load('/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth',map_location='cpu',weights_only=True,mmap=True)
    norms={}
    for layer in (0,15,31):
        key=capture['trace'][f'{layer}.k-raw'].cuda().float()
        beta=state[f'blocks.{layer}.att.k_k'].cuda().float()
        prod=(key*beta).reshape(1,128,40,64)
        norms[str(layer)]=metric(l2_norm(prod),F.normalize(prod,dim=-1))
    tiny=torch.full((1,64),1e-6,device='cuda')
    tiny_result={'torch_normalized_norm':F.normalize(tiny,dim=-1).norm().item(),
                 'fla_default_normalized_norm':l2_norm(tiny).norm().item()}
    result={'original_muon':git('rev-parse','muon'),'before_last_update':git('rev-parse','79d12e8'),
        'current':git('rev-parse','HEAD'),'last_update_rwkv_history':git('log','--oneline','79d12e8..HEAD','--','fla/ops/rwkv7','fla/ops/generalized_delta_rule/dplr').splitlines(),
        'features':features,'tilelang_installed':importlib.util.find_spec('tilelang') is not None,
        'gpu':torch.cuda.get_device_name(),'compute_capability':torch.cuda.get_device_capability(),
        'tilelang_eligibility_only_not_execution':eligibility,'l2_norm_default_formula_difference':norms,
        'l2_norm_tiny_input':tiny_result,'environment_changed':False}
    (PATH/'fla-features-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TILELANG ELIGIBILITY',eligibility,flush=True)
    print('L2 NORMALIZATION FORMULA DIFFERENCE',norms,tiny_result,flush=True)


if __name__=='__main__':main()
