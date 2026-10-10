"""Run upstream fused TimeMix formula/gradient checks at actual RWKV dimensions."""
import importlib.util
import json
from pathlib import Path

import torch
from runtime import ROOT


def main():
    source=ROOT/'3rdparty/flash-linear-attention-260115/tests/ops/test_rwkv7.py'
    spec=importlib.util.spec_from_file_location('upstream_rwkv7_tests',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for dtype in (torch.float32,torch.bfloat16):
        torch.manual_seed(20261009)
        module.test_fused_rwkv7_addcmul(B=2,T=128,H=40,D=64,dtype=dtype,use_g=True)
        rows.append({'test':'fused_addcmul','batch':2,'sequence':128,'hidden':2560,
                     'dtype':str(dtype),'passed':True})
        module.test_fused_k_update(B=2,T=128,H=40,D=64,dtype=dtype,ka_shape=1)
        rows.append({'test':'fused_k_update','batch':2,'sequence':128,'hidden':2560,
                     'dtype':str(dtype),'passed':True})
        print('FUSED PASSED',dtype,flush=True)
    (ROOT/'training-smoke/audit/fused-results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
