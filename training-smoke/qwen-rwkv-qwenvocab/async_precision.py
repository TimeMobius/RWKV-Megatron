"""BF16/FLA full-model sync/async parity with one frozen live teacher response."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import yaml
from megatron.core import parallel_state as ps
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import global_vars

from rwkv_megatron.online_teacher import OnlineTeacherClient, TeacherPrefetchIterator, TeacherTopK, kl_divergence_topk
from rwkv_megatron.rwkv7_layer_specs import get_rwkv7_layer_spec
from rwkv_megatron.rwkv7_model import RWKV7Model

import precision


def main():
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-single.yaml").read_text())
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    global_vars._GLOBAL_ARGS = SimpleNamespace(transformer_impl="local")
    state = torch.load(cfg["load_pretrained_rwkv"], map_location="cpu", weights_only=True)
    dims = cfg["language_model"]
    precision.audit.FULL = True
    precision.audit.DIMS = {"layers": dims["num_layers"], "hidden": dims["hidden_size"],
                            "heads": dims["num_attention_heads"], "ffn": dims["ffn_hidden_size"]}
    config = precision.audit.config_for(1, 1, False, False, fp32=False)
    for name in ("D_AAA_LORA", "D_DECAY_LORA", "D_GATE_LORA", "D_MV_LORA"):
        setattr(config, name, dims[name])
    config.rwkv_key_fusion = True
    model = RWKV7Model(config, transformer_layer_spec=get_rwkv7_layer_spec(),
                      vocab_size=248320, max_sequence_length=128, parallel_output=False,
                      share_embeddings_and_output_weights=False, position_embedding_type="none")
    model = model.cuda().to(config.params_dtype)
    model.from_rwkv_lm(state, False)
    del state
    selected = ("pre_norm.weight", "decoder.final_layernorm.weight")
    for name, param in model.named_parameters():
        param.requires_grad_(name in selected)
    data = IndexedDataset(cfg["data_path"][0])
    cpu_tokens = torch.tensor(data.get(0, length=128).copy(), dtype=torch.long).unsqueeze(0)
    client = OnlineTeacherClient(cfg["rwkv_teacher_url"], cfg["rwkv_teacher_model"], cfg["tokenizer_model"])
    frozen = client.get_logits(cpu_tokens)

    class FrozenTeacher:
        def get_logits(self, input_ids):
            assert torch.equal(input_ids, cpu_tokens)
            return TeacherTopK(frozen.token_ids.clone(), frozen.logprobs.clone())

    tokens = cpu_tokens.cuda()
    positions = torch.arange(128, device="cuda").expand_as(tokens)

    def evaluate(teacher):
        model.zero_grad(set_to_none=True)
        logits = model(tokens, positions, None)
        aligned = TeacherTopK(teacher.token_ids[:, 1:], teacher.logprobs[:, 1:])
        loss = kl_divergence_topk(logits[:, :-1], aligned, torch.ones_like(tokens[:, :-1]))
        loss.backward()
        return logits.detach().cpu(), loss.detach().cpu(), {
            name: param.grad.detach().cpu().clone() for name, param in model.named_parameters()
            if name in selected
        }

    reference = evaluate(frozen)
    prefetch = TeacherPrefetchIterator([{"tokens": cpu_tokens}] * 4, FrozenTeacher(), num_workers=4, lookahead=4)
    try:
        checks = []
        for batch in prefetch:
            actual = evaluate(batch[prefetch.future_key].result())
            checks.append({"logits": precision.metrics(actual[0], reference[0]),
                           "loss_abs": abs(actual[1].item() - reference[1].item()),
                           "gradients": {name: precision.metrics(actual[2][name], reference[2][name])
                                         for name in selected}})
    finally:
        prefetch.close()
    passed = all(x["loss_abs"] == 0 and x["logits"]["max_abs"] == 0
                 and all(g["max_abs"] == 0 and g["finite"] for g in x["gradients"].values()) for x in checks)
    report = {"passed": passed, "checkpoint": cfg["load_pretrained_rwkv"],
              "sequence": 128, "loss": reference[1].item(), "teacher_http_calls": client._calls,
              "checks": checks, "prefetch": prefetch.get_stats(),
              "scope": "Production BF16/FLA model; one frozen real top-100 teacher response; "
                       "compares logits, loss, first/final LayerNorm weight gradients; no optimizer update."}
    Path(__file__).with_name("async-precision-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    dist.destroy_process_group()
    if not passed:
        raise AssertionError("Full-model sync/async parity failed")


if __name__ == "__main__":
    main()
