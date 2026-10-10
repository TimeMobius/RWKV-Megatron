"""Instrument actual eval batch hashes; does not change production training."""
import hashlib
import json
import os
import runpy
from pathlib import Path

import torch
import megatron.training.training as training
from megatron.core.rerun_state_machine import RerunDataIterator

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'training-smoke/audit'/os.environ['FIXED_EVAL_CAPTURE']
events=[];active=None
original_next=RerunDataIterator.__next__
original_evaluate=training.evaluate


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


def captured_evaluate(*args,**kwargs):
    global active
    params=training.get_args()
    active={'train_iteration':params.iteration,'consumed_valid_samples_before':params.consumed_valid_samples,
            'fixed_eval_samples':params.fixed_eval_samples,'is_test':kwargs.get('is_test',False),
            'batch_hashes':[]}
    try:
        result=original_evaluate(*args,**kwargs)
        active['losses']={k:v.item() for k,v in result[0].items()}
        active['consumed_valid_samples_after']=params.consumed_valid_samples
        events.append(active)
        OUTPUT.write_text(json.dumps({'events':events},indent=2)+'\n')
        # Repeat once with unchanged weights to prove fixed samples yield an
        # identical eval value. Restore the accounting counter afterwards.
        if len(events)==1 and params.fixed_eval_samples:
            first=active;before=params.consumed_valid_samples
            active={'replay_same_weights':True,'batch_hashes':[]}
            replay=original_evaluate(*args,**kwargs)
            active['losses']={k:v.item() for k,v in replay[0].items()}
            assert active['batch_hashes']==first['batch_hashes']
            assert active['losses']==first['losses']
            params.consumed_valid_samples=before
            events.append(active)
            OUTPUT.write_text(json.dumps({'events':events},indent=2)+'\n')
        return result
    finally:
        active=None


if __name__=='__main__':
    RerunDataIterator.__next__=capture_next
    training.evaluate=captured_evaluate
    runpy.run_path(str(ROOT/'pretrain_rwkv7.py'),run_name='__main__')
