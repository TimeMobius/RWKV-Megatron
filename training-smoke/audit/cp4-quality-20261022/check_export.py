"""Compare original TP2 checkpoint logits against the exported TP1 evaluation."""

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import yaml

from evaluate import HERE, ROOT, build_model, forward
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
ps.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
model_parallel_cuda_manual_seed(1234)
global_vars._GLOBAL_ARGS = SimpleNamespace(transformer_impl='local')
cfg = yaml.safe_load((ROOT / 'training-smoke/audit/cp4-longrun-20261021/config-train.yaml').read_text())
model = build_model(cfg, tp=2)
path = ROOT / 'training-smoke/audit/cp4-longrun-20261021/checkpoints/iter_0049189' / f'mp_rank_{dist.get_rank():02d}/model_optim_rng.pt'
with torch.inference_mode():
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    ids = json.loads((HERE / 'heldout-samples.json').read_text())['samples'][0]['token_ids'][:-1]
    actual = forward(model, ids)[:, :64].float()
    expected = torch.load(HERE / 'final-calibration.pt', weights_only=True).cuda().float()
    relative = float((actual - expected).norm() / expected.norm())
    agreement = float((actual.argmax(-1) == expected.argmax(-1)).float().mean())
    assert torch.isfinite(actual).all() and relative < 0.01
    exported = torch.load(HERE / 'export/iter_0049189/rwkv_state_dict.pth',
                          map_location='cpu', mmap=True, weights_only=True)
    model.from_rwkv_lm(exported, False)
    reloaded = forward(model, ids)[:, :64].float()
    assert torch.equal(actual, reloaded), 'Same-TP conversion roundtrip changed logits'
    if not dist.get_rank():
        (HERE / 'export-check.json').write_text(json.dumps(dict(
            relative_l2=relative, top1_agreement=agreement, max_abs=float((actual-expected).abs().max()),
            compared_shape=list(actual.shape), tp2_original_vs_tp1_export=True,
            tp2_original_vs_reloaded_export_logits_bitwise_equal=True,
            tolerance_relative_l2=0.01), indent=2) + '\n')
dist.destroy_process_group()
