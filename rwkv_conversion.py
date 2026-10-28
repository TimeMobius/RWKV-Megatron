import argparse
import math
import os
import re
import tempfile
from collections import OrderedDict

import torch

# 定义 tensor parallel 切分维度
tp_split_dim = {
    'embedding.word_embeddings.weight': 0,
    'output_layer.weight': 0,
    'linear_receptance.weight': 0,
    'linear_key.weight': 0,
    'linear_value.weight': 0,
    'linear_proj.weight': 1,
    'linear_up.weight': 0,
    'linear_down.weight': 1,
    'bottleneck_v.bias': 0,
    'bottleneck_learn.bias': 0,
    'bottleneck_decay.bias': 0,
    'rho': 0,
    'beta': 0,
    'alpha': 0,
    'group_norm.weight': 0,
    'group_norm.bias': 0,
    'head.weight': 0,
    'input_layernorm.weight': -1,
    'input_layernorm.bias': -1,
    'pre_mlp_layernorm.weight': -1,
    'pre_mlp_layernorm.bias': -1,
    'final_layernorm.weight': -1,
    'final_layernorm.bias': -1,
    'bottleneck_v.weight_down': -1,
    'bottleneck_v.weight_up': -1,
    'bottleneck_learn.weight_down': -1,
    'bottleneck_learn.weight_up': -1,
    'bottleneck_decay.weight_down': -1,
    'bottleneck_decay.weight_up': -1,
    'bottleneck_gate.weight_down': -1,
    'bottleneck_gate.weight_up': -1,
    'mix_coeff_input': -1,
    'mix_coeffs': -1,
    'pre_norm.weight': -1,
    'pre_norm.bias': -1,
}

def get_split_dim(tensor_name):
    for key in tp_split_dim.keys():
        if tensor_name == key or tensor_name.endswith('.' + key):
            return tp_split_dim[key]
    raise ValueError(f"Unknown tensor name {tensor_name}")

def combine_tp_tensors(key, dim, tensors):
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError(f"Expected tensor shards for {key}")
    if any(tensor.dtype != tensors[0].dtype for tensor in tensors):
        raise ValueError(f"TP dtype mismatch for {key}")
    if dim == -1:
        if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
            raise ValueError(f"Replicated TP tensors differ for {key}")
        return tensors[0].clone()
    if any(tensor.ndim <= dim for tensor in tensors):
        raise ValueError(f"Invalid TP split dimension {dim} for {key}")
    if any(tensor.shape != tensors[0].shape for tensor in tensors):
        raise ValueError(f"TP shape mismatch for {key}")
    return torch.cat(tensors, dim=dim)

def map_megatron_to_rwkv_key(megatron_key, num_layers, norm_offset):
    # 处理嵌入层
    if 'embedding.word_embeddings.weight' in megatron_key:
        return 'emb.weight'
    # 处理输出层
    if 'output_layer.weight' in megatron_key:
        return 'head.weight'
    if 'decoder.final_layernorm.weight' in megatron_key:
        return 'ln_out.weight'
    if 'decoder.final_layernorm.bias' in megatron_key:
        return 'ln_out.bias'
    if 'pre_norm.weight' in megatron_key:
        return 'blocks.0.ln0.weight'
    if 'pre_norm.bias' in megatron_key:
        return 'blocks.0.ln0.bias'
    
    # 提取层编号
    layer_match = re.search(r'layers\.(\d+)\.', megatron_key)
    if not layer_match:
        return None
    layer_num = int(layer_match.group(1))
    prefix = f'blocks.{layer_num}.'
    
    # LayerNorm
    if 'input_layernorm.weight' in megatron_key:
        return prefix + 'ln1.weight'
    if 'input_layernorm.bias' in megatron_key:
        return prefix + 'ln1.bias'
    if 'pre_mlp_layernorm.weight' in megatron_key:
        return prefix + 'ln2.weight'
    if 'pre_mlp_layernorm.bias' in megatron_key:
        return prefix + 'ln2.bias'
    
    # TimeMix (att)
    if 'self_attention.mix_coeff_input' in megatron_key:
        # mix_coeff_input 形状为 [6, hidden_size]，拆分为 6 个 [1, 1, hidden_size]
        channels = ['r', 'w', 'k', 'v', 'a', 'g']
        result = []
        for channel in channels:
            result.append((prefix + f'att.x_{channel}'))
        return result
    if 'self_attention.bottleneck_learn.weight_down' in megatron_key:
        return prefix + 'att.a1'
    if 'self_attention.bottleneck_learn.weight_up' in megatron_key:
        return prefix + 'att.a2'
    if 'self_attention.bottleneck_learn.bias' in megatron_key:
        return prefix + 'att.a0'
    if 'self_attention.bottleneck_decay.weight_down' in megatron_key:
        return prefix + 'att.w1'
    if 'self_attention.bottleneck_decay.weight_up' in megatron_key:
        return prefix + 'att.w2'
    if 'self_attention.bottleneck_decay.bias' in megatron_key:
        return prefix + 'att.w0'
    if 'self_attention.bottleneck_gate.weight_down' in megatron_key:
        return prefix + 'att.g1'
    if 'self_attention.bottleneck_gate.weight_up' in megatron_key:
        return prefix + 'att.g2'
    if 'self_attention.rho' in megatron_key:
        return prefix + 'att.r_k'
    if 'self_attention.beta' in megatron_key:
        return prefix + 'att.k_k'
    if 'self_attention.alpha' in megatron_key:
        return prefix + 'att.k_a'
    if 'self_attention.linear_receptance.weight' in megatron_key:
        return prefix + 'att.receptance.weight'
    if 'self_attention.linear_key.weight' in megatron_key:
        return prefix + 'att.key.weight'
    if 'self_attention.linear_value.weight' in megatron_key:
        return prefix + 'att.value.weight'
    if 'self_attention.linear_proj.weight' in megatron_key:
        return prefix + 'att.output.weight'
    if 'self_attention.group_norm.weight' in megatron_key:
        return prefix + 'att.ln_x.weight'
    if 'self_attention.group_norm.bias' in megatron_key:
        return prefix + 'att.ln_x.bias'
    if 'self_attention.bottleneck_v.weight_down' in megatron_key:
        if layer_num == 0:
            return None
        else:
            return prefix + 'att.v1'
    if 'self_attention.bottleneck_v.weight_up' in megatron_key:
        if layer_num == 0:
            return None
        else:
            return prefix + 'att.v2'
    if 'self_attention.bottleneck_v.bias' in megatron_key:
        if layer_num == 0:
            return None
        else:
            return prefix + 'att.v0'
    
    # ChannelMix (ffn)
    if 'mlp.mix_coeffs' in megatron_key:
        return prefix + 'ffn.x_k'
    if 'mlp.linear_up.weight' in megatron_key:
        return prefix + 'ffn.key.weight'
    if 'mlp.linear_down.weight' in megatron_key:
        return prefix + 'ffn.value.weight'
    
    return None

def resolve_checkpoint(load_dir, iteration=None):
    load_dir = os.path.abspath(os.fspath(load_dir))
    match = re.fullmatch(r'iter_(\d+)', os.path.basename(load_dir))
    if match:
        directory_iteration = int(match.group(1))
        if iteration is not None and iteration != directory_iteration:
            raise ValueError(f"--iteration {iteration} conflicts with {load_dir}")
        iteration = directory_iteration
        input_model_dir = load_dir
    else:
        if iteration is None:
            tracker_filename = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
            with open(tracker_filename) as tracker:
                try:
                    iteration = int(tracker.read().strip())
                except ValueError as error:
                    raise ValueError(f"Invalid iteration in {tracker_filename}") from error
        input_model_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
    if iteration < 0:
        raise ValueError("Checkpoint iteration must be nonnegative")
    if not os.path.isdir(input_model_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {input_model_dir}")
    return input_model_dir, iteration


def pipeline_layer_counts(config):
    """Resolve ordinary pipeline endpoint overrides from config or saved args."""
    pp, total = config.pipeline_model_parallel_size, config.num_layers
    endpoints = []
    for side in ('first', 'last'):
        names = (f'num_layers_in_{side}_pipeline_stage', f'decoder_{side}_pipeline_num_layers')
        values = [getattr(config, name, None) for name in names]
        specified = [value for value in values if value is not None]
        if len(specified) == 2 and specified[0] != specified[1]:
            raise ValueError(f'Conflicting {side} pipeline layer counts')
        value = specified[0] if specified else None
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'Invalid {side} pipeline layer count')
        endpoints.append(value)
    if pp == 1:
        if any(value is not None for value in endpoints):
            raise ValueError('Pipeline endpoint overrides require PP > 1')
        return [total]
    remaining_stages = pp - sum(value is not None for value in endpoints)
    remaining_layers = total - sum(value for value in endpoints if value is not None)
    if remaining_stages == 0:
        if remaining_layers:
            raise ValueError('Pipeline endpoint layers do not cover the model')
        return endpoints
    if remaining_layers < remaining_stages or remaining_layers % remaining_stages:
        raise ValueError('Remaining pipeline layers must divide evenly across remaining stages')
    counts = [remaining_layers // remaining_stages] * pp
    for index, value in zip((0, -1), endpoints):
        if value is not None:
            counts[index] = value
    return counts


def validate_model_config(config):
    for name in ('tensor_model_parallel_size', 'pipeline_model_parallel_size',
                 'num_layers', 'hidden_size', 'num_attention_heads'):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid model config {name}={value}")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if config.num_attention_heads % config.tensor_model_parallel_size:
        raise ValueError("num_attention_heads must be divisible by tensor_model_parallel_size")
    pipeline_layer_counts(config)
    if any(getattr(config, name, None) is not None for name in (
        'virtual_pipeline_model_parallel_size',
        'pipeline_model_parallel_layout',
    )):
        raise ValueError("Virtual or explicit pipeline layouts are not supported")


def resolve_norm_offset(config, norm_offset=None):
    if norm_offset is None:
        norm_offset = 1.0 if getattr(config, 'layernorm_zero_centered_gamma', False) else 0.0
    if not math.isfinite(norm_offset):
        raise ValueError("--norm-offset must be finite")
    return norm_offset


def convert_pipeline_stage(tp_models, pp_rank, config, norm_offset):
    if len(tp_models) != config.tensor_model_parallel_size:
        raise ValueError(f"Wrong number of TP shards for pipeline rank {pp_rank}")
    if any(set(model) != set(tp_models[0]) for model in tp_models[1:]):
        raise ValueError(f"TP key mismatch for pipeline rank {pp_rank}")
    layer_counts = pipeline_layer_counts(config)
    layers_per_stage = layer_counts[pp_rank]
    layers_start = sum(layer_counts[:pp_rank])
    state_dict = OrderedDict()
    for source_key in tp_models[0]:
        if source_key.endswith('._extra_state'):
            continue
        layer_match = re.search(r'layers\.(\d+)\.', source_key)
        megatron_key = source_key
        if layer_match:
            local_layer = int(layer_match.group(1))
            if local_layer >= layers_per_stage:
                raise ValueError(f"Invalid local pipeline layer in {source_key}")
            global_layer = local_layer + layers_start
            megatron_key = re.sub(r'layers\.\d+\.', f'layers.{global_layer}.', source_key, count=1)
        rwkv_key = map_megatron_to_rwkv_key(megatron_key, config.num_layers, norm_offset)
        if rwkv_key is None:
            if megatron_key in {
                'decoder.layers.0.self_attention.bottleneck_v.weight_down',
                'decoder.layers.0.self_attention.bottleneck_v.weight_up',
                'decoder.layers.0.self_attention.bottleneck_v.bias',
            }:
                continue
            raise ValueError(f"Unmapped checkpoint parameter: {megatron_key}")
        tensor = combine_tp_tensors(
            source_key, get_split_dim(source_key), [model[source_key] for model in tp_models]
        ).detach().contiguous()
        if isinstance(rwkv_key, list):
            if tuple(tensor.shape) != (6, config.hidden_size):
                raise ValueError(f"Invalid mix_coeff_input shape for {megatron_key}: {tensor.shape}")
            mapped = [
                (key, tensor[index:index + 1].reshape(1, 1, config.hidden_size).clone())
                for index, key in enumerate(rwkv_key)
            ]
        else:
            if rwkv_key.endswith('.att.r_k'):
                tensor = tensor.reshape(config.num_attention_heads,
                                        config.hidden_size // config.num_attention_heads)
            if rwkv_key.endswith(('.att.k_k', '.att.k_a', '.att.v0', '.att.a0', '.att.w0', '.ffn.x_k')):
                tensor = tensor.reshape(1, 1, -1)
            if rwkv_key == 'ln_out.weight' or re.fullmatch(r'blocks\.\d+\.ln[012]\.weight', rwkv_key):
                tensor = tensor + norm_offset
            mapped = [(rwkv_key, tensor)]
        for key, value in mapped:
            if key in state_dict:
                raise ValueError(f"Duplicate output parameter: {key}")
            state_dict[key] = value
    return state_dict


def expected_rwkv_shapes(config, state_dict):
    hidden_size = config.hidden_size
    heads = config.num_attention_heads
    embedding = state_dict.get('emb.weight')
    vocab_size = getattr(config, 'padded_vocab_size', None)
    if vocab_size is None:
        if embedding is None or embedding.ndim != 2:
            raise ValueError("Missing or invalid emb.weight")
        vocab_size = embedding.shape[0]
    ffn_size = getattr(config, 'ffn_hidden_size', None)
    if ffn_size is None:
        ffn = state_dict.get('blocks.0.ffn.key.weight')
        if ffn is None or ffn.ndim != 2:
            raise ValueError("Missing or invalid blocks.0.ffn.key.weight")
        ffn_size = ffn.shape[0]
    shapes = {
        'emb.weight': (vocab_size, hidden_size),
        'head.weight': (vocab_size, hidden_size),
        'ln_out.weight': (hidden_size,), 'ln_out.bias': (hidden_size,),
        'blocks.0.ln0.weight': (hidden_size,), 'blocks.0.ln0.bias': (hidden_size,),
    }
    for layer in range(config.num_layers):
        prefix = f'blocks.{layer}.'
        suffix_shapes = {
            'ln1.weight': (hidden_size,), 'ln1.bias': (hidden_size,),
            'ln2.weight': (hidden_size,), 'ln2.bias': (hidden_size,),
            'att.r_k': (heads, hidden_size // heads),
            'att.ln_x.weight': (hidden_size,), 'att.ln_x.bias': (hidden_size,),
            'ffn.x_k': (1, 1, hidden_size),
            'ffn.key.weight': (ffn_size, hidden_size),
            'ffn.value.weight': (hidden_size, ffn_size),
        }
        for channel in 'rwkvag':
            suffix_shapes[f'att.x_{channel}'] = (1, 1, hidden_size)
        for suffix in ('k_k', 'k_a', 'w0', 'a0'):
            suffix_shapes[f'att.{suffix}'] = (1, 1, hidden_size)
        for projection in ('receptance', 'key', 'value', 'output'):
            suffix_shapes[f'att.{projection}.weight'] = (hidden_size, hidden_size)
        for family, config_name in (
            ('w', 'D_DECAY_LORA'), ('a', 'D_AAA_LORA'),
            ('g', 'D_GATE_LORA'), ('v', 'D_MV_LORA'),
        ):
            if family == 'v' and layer == 0:
                continue
            rank = getattr(config, config_name, None)
            if rank is None:
                down = state_dict.get(prefix + f'att.{family}1')
                if down is None or down.ndim != 2:
                    raise ValueError(f"Missing or invalid {prefix}att.{family}1")
                rank = down.shape[1]
            if rank <= 0:
                raise ValueError(f"Invalid bottleneck rank for {prefix}att.{family}1")
            suffix_shapes[f'att.{family}1'] = (hidden_size, rank)
            suffix_shapes[f'att.{family}2'] = (rank, hidden_size)
            if family == 'v':
                suffix_shapes['att.v0'] = (1, 1, hidden_size)
        shapes.update((prefix + suffix, shape) for suffix, shape in suffix_shapes.items())
    return shapes


def validate_rwkv_state_dict(state_dict, config):
    shapes = expected_rwkv_shapes(config, state_dict)
    missing = sorted(set(shapes) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(shapes))
    if missing or unexpected:
        raise ValueError(f"Invalid RWKV keys: missing={missing}, unexpected={unexpected}")
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"Expected floating-point tensor for {key}")
        if tuple(tensor.shape) != shapes[key]:
            raise ValueError(f"Invalid shape for {key}: {tuple(tensor.shape)}, expected {shapes[key]}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite values in {key}")


def atomic_save(state_dict, save_file):
    descriptor, temporary_file = tempfile.mkstemp(prefix='.rwkv-', suffix='.pth',
                                                dir=os.path.dirname(save_file))
    os.close(descriptor)
    try:
        torch.save(state_dict, temporary_file)
        os.replace(temporary_file, save_file)
    finally:
        if os.path.exists(temporary_file):
            os.unlink(temporary_file)


def main(args):
    input_model_dir, iteration = resolve_checkpoint(args.load_dir, getattr(args, 'iteration', None))
    input_sub_models = sorted(
        entry for entry in os.listdir(input_model_dir)
        if re.fullmatch(r'mp_rank_\d+(?:_\d+)?', entry)
        and os.path.isdir(os.path.join(input_model_dir, entry))
    )
    if not input_sub_models:
        raise ValueError(f"No mp_rank_* checkpoint shards found under {input_model_dir}")
    sample_file = os.path.join(input_model_dir, input_sub_models[0], 'model_optim_rng.pt')
    sample = torch.load(sample_file, map_location='cpu', weights_only=False)
    config = sample['args']
    del sample
    validate_model_config(config)
    norm_offset = resolve_norm_offset(config, getattr(args, 'norm_offset', None))
    print(f"Converting {input_model_dir}: TP={config.tensor_model_parallel_size}, "
          f"PP={config.pipeline_model_parallel_size}, layers={config.num_layers}, "
          f"head_shape=({config.num_attention_heads}, {config.hidden_size // config.num_attention_heads}), "
          f"norm_offset={norm_offset}")
    state_dict = OrderedDict()
    config_fields = (
        'tensor_model_parallel_size', 'pipeline_model_parallel_size', 'num_layers',
        'hidden_size', 'num_attention_heads', 'ffn_hidden_size', 'padded_vocab_size',
        'layernorm_zero_centered_gamma', 'D_DECAY_LORA', 'D_AAA_LORA', 'D_GATE_LORA', 'D_MV_LORA',
        'num_layers_in_first_pipeline_stage', 'num_layers_in_last_pipeline_stage',
        'decoder_first_pipeline_num_layers', 'decoder_last_pipeline_num_layers',
    )
    for pp_rank in range(config.pipeline_model_parallel_size):
        tp_models = []
        for tp_rank in range(config.tensor_model_parallel_size):
            rank_dir = f'mp_rank_{tp_rank:02d}'
            if config.pipeline_model_parallel_size > 1:
                rank_dir += f'_{pp_rank:03d}'
            model_file = os.path.join(input_model_dir, rank_dir, 'model_optim_rng.pt')
            model_data = torch.load(model_file, map_location='cpu', weights_only=False)
            if model_data.get('iteration', iteration) != iteration:
                raise ValueError(f"Checkpoint iteration mismatch in {model_file}")
            if any(getattr(model_data['args'], name, None) != getattr(config, name, None)
                   for name in config_fields):
                raise ValueError(f"Checkpoint config mismatch in {model_file}")
            tp_models.append(model_data['model'])
            del model_data
        stage = convert_pipeline_stage(tp_models, pp_rank, config, norm_offset)
        overlap = set(state_dict) & set(stage)
        if overlap:
            raise ValueError(f"Duplicate output parameters across pipeline stages: {sorted(overlap)}")
        state_dict.update(stage)
        del stage, tp_models
        print(f"Converted pipeline rank {pp_rank}")
    validate_rwkv_state_dict(state_dict, config)
    save_dir = os.path.join(args.save_dir, f'iter_{iteration:07d}')
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, 'rwkv_state_dict.pth')
    atomic_save(state_dict, save_file)
    with open(os.path.join(args.save_dir, 'latest_checkpointed_iteration.txt'), 'w') as tracker:
        tracker.write(str(iteration))
    print(f"Validated {len(state_dict)} tensors; converted RWKV state dict saved to {save_file}")
    return save_file


def build_parser():
    parser = argparse.ArgumentParser(description="Convert Megatron RWKV7 checkpoints to native RWKV weights")
    parser.add_argument('--load-dir', required=True, help="检查点根目录或指定的 iter_* 目录")
    parser.add_argument('--save-dir', required=True, help="输出检查点目录")
    parser.add_argument('--iteration', type=int, help="指定迭代，覆盖根目录的 latest tracker")
    parser.add_argument('--norm-offset', type=float, default=None,
                        help="LayerNorm gamma 偏移；默认按 zero-centered gamma 自动选择 +1 或 0")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
