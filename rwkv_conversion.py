import copy
import os
import re
import shutil
from collections import OrderedDict

import torch
import argparse

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
    'layernorm.weight': -1,
    'layernorm.bias': -1,
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
        if key in tensor_name:
            return tp_split_dim[key]
    raise Exception(f"Unknown tensor name {tensor_name}")

def combine_tp_tensors(key, dim, tensors):
    if dim == -1:
        # [修改 1] 这里添加 clone，防止引用原始大块内存，便于 GC 回收
        return tensors[0].clone()
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

def main(args):
    print("\n====RUNNING RWKV CHECKPOINT CONVERSION (With Compaction)====\n")

    # 读取最新迭代
    tracker_filename = os.path.join(args.load_dir, 'latest_checkpointed_iteration.txt')
    try:
        with open(tracker_filename, 'r') as f:
            metastring = f.read().strip()
            iteration = int(metastring)
    except FileNotFoundError:
        raise Exception(f"Tracker file {tracker_filename} not found")

    # 加载样本模型（仅提取 args 和 model）
    input_model_dir = os.path.join(args.load_dir, f'iter_{iteration:07d}')
    input_sub_models = os.listdir(input_model_dir)
    sample_model_file = os.path.join(input_model_dir, input_sub_models[0], "model_optim_rng.pt")
    try:
        sample_model = torch.load(sample_model_file, map_location='cpu', weights_only=False)  # 加载整个检查点
        sample_args = sample_model['args']
        sample_model_weights = sample_model['model']
        del sample_model  # 释放内存
    except Exception as e:
        raise Exception(f"Failed to load sample model {sample_model_file}: {str(e)}")
    print(f"Sample model {sample_model_file} is loaded.\n")

    # 获取输入 TP 和 PP 大小
    input_tp_rank = sample_args.tensor_model_parallel_size
    input_pp_rank = sample_args.pipeline_model_parallel_size
    num_layers = sample_args.num_layers
    head_num = sample_args.num_attention_heads
    head_dim = sample_args.hidden_size // head_num
    num_layers_per_pipeline_rank = num_layers // input_pp_rank
    print(f"Model config: TP={input_tp_rank}, PP={input_pp_rank}, num_layers={num_layers}, num_layers_per_pipeline_rank={num_layers_per_pipeline_rank}")

    # 构建完整模型
    full_model = OrderedDict()
    for pp in range(input_pp_rank):
        print(f"[INFO] Processing input pipeline rank {pp}")
        tp_models = []
        for tp in range(input_tp_rank):
            dir_name = f"mp_rank_{tp:02d}"
            if input_pp_rank > 1:
                dir_name += f"_{pp:03d}"
            model_file = os.path.join(input_model_dir, dir_name, "model_optim_rng.pt")
            try:
                model_data = torch.load(model_file, map_location='cpu', weights_only=False)
                tp_models.append(model_data['model'])
                del model_data  # 释放内存
            except Exception as e:
                raise Exception(f"Failed to load model {model_file}: {str(e)}")
            print(f"Model {model_file} is loaded.")

        # 合并 TP 分片
        combined_tp_model = OrderedDict()
        for key, original_tensor in tp_models[0].items():
            if "_extra_state" in key:
                combined_tp_model[key] = original_tensor
                continue

            split_dim = get_split_dim(key)
            print(f"key: {key}, shape: {tp_models[0][key].shape}, split_dim: {split_dim}")
            if split_dim != -1 and input_tp_rank > 1:
                try:
                    combined_tensor = combine_tp_tensors(
                        key, split_dim,
                        [tp_models[j][key].cpu() for j in range(input_tp_rank)] # 如果是被切分的模型，那么tp_models[1]是和tp_models[0]不一样的，否则一样
                    )
                    combined_tp_model[key] = combined_tensor
                    print(f"Combined {key}: {combined_tensor.shape}")
                except Exception as e:
                    raise Exception(f"Failed to combine TP tensors for {key}: {str(e)}")
            else:
                # [修改 2] 使用 .clone() 确保切断与原始大文件的联系
                combined_tp_model[key] = original_tensor.clone()

        # 重新映射层编号
        for key, tensor in combined_tp_model.items():
            try:
                layer_num = int(re.search(r'layers\.(\d+)\.', key).group(1))
                new_key = key.replace(f'layers.{layer_num}', f'layers.{layer_num + pp * num_layers_per_pipeline_rank}')
            except:
                new_key = key
            full_model[new_key] = tensor
        del tp_models  # 释放内存
    print("\n[INFO] Loaded combined model\n")

    # 转换为 RWKV 原始格式
    rwkv_state_dict = OrderedDict()
    for megatron_key, tensor in full_model.items():
        if "_extra_state" in megatron_key:
            continue
        rwkv_key = map_megatron_to_rwkv_key(megatron_key, num_layers, args.norm_offset)
        if rwkv_key is None:
            print(f"Skipping unmapped key: {megatron_key}")
            continue

        # 特殊处理
        if isinstance(rwkv_key, list):
            for i, key in enumerate(rwkv_key):
                # [修改 3] 添加 .clone().detach().contiguous() 确保内存连续且无额外 Storage 占用
                rwkv_state_dict[key] = tensor[i:i+1].view(1, 1, -1).clone().detach().contiguous()
            # 注意：这里我们取 list 中的一个 key 来打印 shape 提示，这只是为了打印信息不报错
            print(f"Mapped {megatron_key} -> {rwkv_key}: {rwkv_state_dict[rwkv_key[0]].shape}")
            continue
        
        if 'r_k' in rwkv_key:
            tensor = tensor.view(head_num, head_dim)  # 转换为 [head_num, head_dim]
        if any(x in rwkv_key for x in ['k_k', 'k_a', 'v0', 'a0', 'w0', "x_k"]):
            tensor = tensor.view(1, 1, -1)  # 转换为 [1, 1, 5120]
        
        # [修改 4] 添加 .clone().detach().contiguous() 确保内存连续且无额外 Storage 占用
        rwkv_state_dict[rwkv_key] = tensor.clone().detach().contiguous()
        print(f"Mapped {megatron_key} -> {rwkv_key}: {tensor.shape}")

    # 保存转换后的权重
    save_dir = os.path.join(args.save_dir, f'iter_{iteration:07d}')
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "rwkv_state_dict.pth")
    torch.save(rwkv_state_dict, save_file)
    print(f"Converted RWKV state dict saved to {save_file}")

    # 验证输出权重
    expected_keys = [f'blocks.{i}.att.r_k' for i in range(num_layers)] + \
                    [f'blocks.{i}.ffn.key.weight' for i in range(num_layers)] + \
                    ['emb.weight', 'head.weight']
    missing_keys = [key for key in expected_keys if key not in rwkv_state_dict]
    if missing_keys:
        print(f"Warning: Missing keys in output state_dict: {missing_keys}")

    # 更新 tracker 文件
    tracker_filename_out = os.path.join(args.save_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename_out, 'w') as f:
        f.write(str(iteration))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, required=True, help="输入检查点目录")
    parser.add_argument('--save-dir', type=str, required=True, help="输出检查点目录")
    parser.add_argument('--norm-offset', type=float, default=0.0, help="LayerNorm 偏移（用于 zero-centered gamma）")

    args = parser.parse_args()
    main(args)