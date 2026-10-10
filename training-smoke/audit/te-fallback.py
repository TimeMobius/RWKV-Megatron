"""Exercise RWKV7 with TE available, too old, or absent in a fresh process."""
import argparse
import builtins
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('te', 'old', 'missing'), required=True)
args = parser.parse_args()
if args.mode == 'missing':
    original_import = builtins.__import__
    def import_without_te(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == 'transformer_engine_torch' or name == 'transformer_engine' or name.startswith('transformer_engine.')):
            raise ModuleNotFoundError('TE intentionally unavailable for fallback test', name=name)
        return original_import(name, globals, locals, fromlist, level)
    builtins.__import__ = import_without_te
    original_find_spec = importlib.util.find_spec
    def find_without_te(name, package=None):
        if name == 'transformer_engine' or name == 'transformer_engine_torch' or name.startswith('transformer_engine.'):
            return None
        return original_find_spec(name, package)
    importlib.util.find_spec = find_without_te

import torch
import torch.distributed as dist
from types import SimpleNamespace
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import global_vars
from rwkv_megatron import normalization
from rwkv_megatron import rwkv7_layer_specs as specs
from rwkv_megatron.rwkv7_model import RWKV7Model
from rwkv_conversion import convert_pipeline_stage, validate_rwkv_state_dict

PATH = Path(__file__).resolve().parent
if args.mode == 'old':
    normalization.is_te_min_version = lambda version: False
torch.cuda.set_device(0)
dist.init_process_group('nccl')
ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
model_parallel_cuda_manual_seed(1234)
global_vars._GLOBAL_ARGS = SimpleNamespace(transformer_impl='local')
cfg = TransformerConfig(num_layers=4, hidden_size=256, num_attention_heads=4, ffn_hidden_size=1024,
    kv_channels=64, bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
    hidden_dropout=0., attention_dropout=0., add_bias_linear=False, add_qkv_bias=False,
    perform_initialization=False, gradient_accumulation_fusion=False,
    persist_layer_norm=False, layernorm_epsilon=1e-5)
for name in ('D_AAA_LORA', 'D_DECAY_LORA', 'D_GATE_LORA', 'D_MV_LORA'):
    setattr(cfg, name, 16)
state = torch.load(PATH / 'cropped-native.pth', map_location='cpu', weights_only=True)
model = RWKV7Model(cfg, transformer_layer_spec=specs.get_rwkv7_layer_spec(), vocab_size=128,
    max_sequence_length=128, pre_process=True, post_process=True, parallel_output=False,
    share_embeddings_and_output_weights=False, position_embedding_type='none').cuda().to(torch.bfloat16)
model.from_rwkv_lm(state, False)
norms = [model.pre_norm, model.decoder.final_layernorm]
for layer in model.decoder.layers:
    norms.extend((layer.input_layernorm, layer.pre_mlp_layernorm))
types = [type(m).__module__ + '.' + type(m).__name__ for m in norms]
expected = 'transformer_engine.pytorch.module.layernorm.LayerNorm' if args.mode == 'te' else 'megatron.core.fusions.fused_layer_norm.FusedLayerNorm'
assert len(norms) == 10 and all(t == expected for t in types), types
torch.manual_seed(20261009)
tokens = torch.randint(1, 128, (2, 128), device='cuda')
labels = torch.randint(1, 128, (2, 128), device='cuda')
logits = model(tokens, torch.arange(128, device='cuda').expand_as(tokens), None)
loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 128), labels.reshape(-1))
loss.backward()
assert torch.isfinite(logits).all() and torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert model.decoder.layers[0].self_attention.linear_value.weight.grad.float().norm() > 0
native = convert_pipeline_stage([model.state_dict()], 0, cfg, 0)
validate_rwkv_state_dict(native, cfg)
cfg.sequence_parallel = True
norm = normalization.get_rwkv7_layernorm_impl()(cfg, 256, 1e-5).cuda().to(torch.bfloat16)
assert norm.weight.sequence_parallel and norm.bias.sequence_parallel
cfg.layernorm_zero_centered_gamma = True
centered = normalization.get_rwkv7_layernorm_impl()(cfg, 256, 1e-5).cuda().to(torch.bfloat16)
with torch.no_grad():
    centered.weight.fill_(0.25)
    centered.bias.fill_(0.125)
x = torch.randn(8, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
y = centered(x)
reference = torch.nn.functional.layer_norm(x.float(), (256,), centered.weight.float() + 1,
                                         centered.bias.float(), 1e-5)
error = (y.float() - reference).norm() / reference.norm()
assert error < 0.01
y.sum().backward()
assert torch.isfinite(x.grad).all()
result = dict(mode=args.mode, norm_types=types, loss=loss.item(), finite_forward_backward=True,
    converter_validated=True, native_tensors=len(native), sequence_parallel_param_tags=True,
    zero_centered_gamma_relative_l2=error.item(), te_imported=specs.HAVE_TE,
    first_value_weight_gradient_norm=model.decoder.layers[0].self_attention.linear_value.weight.grad.float().norm().item())
(PATH / f'te-fallback-{args.mode}.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
dist.destroy_process_group()
