"""Use the existing all-gradient real-PP audit with a frozen old/new transport."""
import importlib.util
import os
from pathlib import Path

PATH=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('payload_audit',PATH.parent/'multigpu-20261012/distributed.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
if os.environ.get('V0_PAYLOAD_BEFORE')=='1':
    from legacy import install
    install()
audit.OUT=PATH/os.environ['V0_CASE_TAG'];audit.OUT.mkdir(exist_ok=True)
original=audit.config_for


def config(*args,**kwargs):
    cfg=original(*args,**kwargs)
    cfg.rwkv_key_fusion=os.environ.get('V0_KEY_FUSION','0')=='1'
    if os.environ.get('V0_SELECTIVE_NORM')=='1':
        cfg.recompute_granularity='selective';cfg.recompute_modules=['layernorm']
        cfg.recompute_method=None;cfg.recompute_num_layers=None
    return cfg


audit.config_for=config
if __name__=='__main__':audit.main()
