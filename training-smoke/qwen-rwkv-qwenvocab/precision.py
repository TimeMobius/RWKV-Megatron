"""Full small-model formula audit and live teacher KL/gradient comparison.

The FP32 audit replaces only WKV with the existing Torch recurrence oracle.
BF16 executes production FLA and compares its KL value/logit gradient with the
source distiller, using one frozen HTTP teacher response for both losses.
"""

import gc
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import yaml
from megatron.core import parallel_state as ps
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars

from rwkv_megatron.online_teacher import OnlineTeacherClient, TeacherTopK, kl_divergence_topk
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training-smoke/audit"))
import distributed as audit
from native_reference import NativeReference


def metrics(a, b):
    a, b = a.detach().float(), b.detach().float()
    return {"max_abs": (a-b).abs().max().item(),
            "relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def main():
    cfg = yaml.safe_load((ROOT / "configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-single.yaml").read_text())
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    global_vars._GLOBAL_ARGS = SimpleNamespace(transformer_impl="local")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(cfg["load_pretrained_rwkv"], map_location="cpu", weights_only=True)
    data = IndexedDataset(cfg["data_path"][0])
    tokens = torch.tensor(data.get(0, length=64).copy(), dtype=torch.long).unsqueeze(0).cuda()
    teacher = OnlineTeacherClient(cfg["rwkv_teacher_url"], cfg["rwkv_teacher_model"], cfg["tokenizer_model"]).get_logits(tokens)
    aligned = TeacherTopK(teacher.token_ids[:, 1:], teacher.logprobs[:, 1:])
    mask = torch.ones_like(tokens[:, :-1], dtype=torch.bool)
    report = {"checkpoint": cfg["load_pretrained_rwkv"], "data": cfg["data_path"][0],
              "tokens": tokens.cpu().tolist(), "sequence": 64, "batch": 1,
              "checkpoint_parameters": sum(x.numel() for x in state.values()),
              "limits": ["FP32 formula check uses Torch WKV, not the production BF16 kernel.",
                         "KL parity uses the source loss and fixed teacher output; it does not certify model quality."]}
    selected = ("pre_norm.weight", "decoder.final_layernorm.weight")
    native_names = ("blocks.0.ln0.weight", "ln_out.weight")

    def build(fp32):
        audit.FULL = True
        dims = cfg["language_model"]
        audit.DIMS = {"layers": dims["num_layers"], "hidden": dims["hidden_size"],
                      "heads": dims["num_attention_heads"], "ffn": dims["ffn_hidden_size"]}
        c = audit.config_for(1, 1, False, False, fp32=fp32)
        for name in ("D_AAA_LORA", "D_DECAY_LORA", "D_GATE_LORA", "D_MV_LORA"):
            setattr(c, name, dims[name])
        c.rwkv_key_fusion = True
        model = RWKV7Model(c, transformer_layer_spec=get_rwkv7_layer_spec(),
                          vocab_size=248320, max_sequence_length=64, parallel_output=False,
                          share_embeddings_and_output_weights=False, position_embedding_type="none")
        model = model.cuda().to(c.params_dtype)
        model.from_rwkv_lm(state, False)
        for name, p in model.named_parameters():
            p.requires_grad_(name in selected)
        return model

    model = build(True)
    for layer in model.decoder.layers:
        layer.self_attention.core_attention.forward = audit.recurrence
    logits = model(tokens, torch.arange(64, device="cuda").expand_as(tokens), None)
    loss = kl_divergence_topk(logits[:, :-1], aligned, mask)
    loss.backward()
    gradients = {name: p.grad.cpu().clone() for name, p in model.named_parameters() if name in selected}
    actual_logits, actual_loss = logits.detach().cpu(), loss.item()
    del logits, loss, model
    gc.collect()
    torch.cuda.empty_cache()

    reference = NativeReference(state, torch.float32)
    reference.mark = lambda key, value: value
    for key, p in reference.weights.items():
        p.requires_grad_(key.replace("__", ".") in native_names)
    logits = reference(tokens)
    loss = kl_divergence_topk(logits[:, :-1], aligned, mask)
    loss.backward()
    formula = {"logits": metrics(actual_logits, logits.cpu()), "loss_abs": abs(actual_loss-loss.item()),
               "gradients": {key: metrics(gradients[key], reference.w(native).grad.cpu())
                             for key, native in zip(selected, native_names)}}
    report["fp32_formula"] = formula
    del reference, logits, loss
    gc.collect()
    torch.cuda.empty_cache()

    model = build(False)
    logits = model(tokens, torch.arange(64, device="cuda").expand_as(tokens), None).detach()
    source_path = Path("/mnt/data/Codes/RWKV/qwen3_5_to_rwkv7_distillation/src/distillation_loss.py")
    spec = importlib.util.spec_from_file_location("distill_source_loss", source_path)
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)
    a = logits.float().requires_grad_()
    b = logits.float().clone().requires_grad_()
    actual = kl_divergence_topk(a[:, :-1], aligned, mask)
    expected = source.DistillationLoss(temperature=2.0).kl_divergence_topk(
        b, teacher.token_ids, teacher.logprobs, torch.ones_like(tokens), shift_targets=True)
    ga, = torch.autograd.grad(actual, a)
    gb, = torch.autograd.grad(expected, b)
    report["production_bf16_source_kl"] = {"loss": actual.item(), "source_loss": expected.item(),
                                          "loss_abs": abs(actual.item()-expected.item()),
                                          "logit_gradient": metrics(ga, gb)}
    passed = (formula["logits"]["relative_l2"] < 1e-5 and formula["loss_abs"] < 1e-5
              and all(x["relative_l2"] < 1e-3 and x["finite"] for x in formula["gradients"].values())
              and report["production_bf16_source_kl"]["loss_abs"] < 1e-5
              and report["production_bf16_source_kl"]["logit_gradient"]["relative_l2"] < 1e-5)
    report["passed"] = passed
    (Path(__file__).parent / "precision-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    dist.destroy_process_group()
    if not passed:
        raise AssertionError("Precision check failed; see precision-results.json")


if __name__ == "__main__":
    main()
