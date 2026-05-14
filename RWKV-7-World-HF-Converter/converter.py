import torch
import os
import json
import argparse
from safetensors.torch import save_file

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='转换RWKV模型格式为safetensors')
parser.add_argument('--source', type=str, required=True, help='源模型路径（pt/bin）')
parser.add_argument('--target', type=str, required=True, help='目标模型路径（safetensors）') # model.safetensors
parser.add_argument('--config', type=str, help='可选，需同步更新的config.json路径')
args = parser.parse_args()

SOURCE_MODEL = args.source
TARGET_MODEL = args.target
CONFIG_PATH = args.config

# 如果目标文件存在，则删除
if os.path.exists(TARGET_MODEL):
    os.remove(TARGET_MODEL)

# 加载源模型
model = torch.load(SOURCE_MODEL, mmap=True, map_location='cpu')

# 打印原始模型的权重键
print("\n=== 原始模型权重键 ===")
keys_list = sorted(list(model.keys()))
print(f"模型中共有 {len(keys_list)} 个权重键")
print("\n权重键列表：")
print(keys_list)
# 打印每个权重的形状
print("\n=== 原始模型权重形状 ===")
for key in keys_list:
    shape = model[key].shape
    print(f"'{key}': {shape}")

# 动态检测模型的层数
block_keys = [k for k in keys_list if k.startswith("blocks.")]
num_blocks = max(int(k.split('.')[1]) for k in block_keys) + 1 if block_keys else 0
print(f"\n检测到模型有 {num_blocks} 层")

new_model = {}


def find_first_shape(model_state, pattern, total_blocks):
    for idx in range(total_blocks):
        key = pattern.format(idx=idx)
        if key in model_state:
            return model_state[key].shape
    return None


def infer_hidden_size(model_state, total_blocks):
    if "emb.weight" in model_state and model_state["emb.weight"].ndim == 2:
        return int(model_state["emb.weight"].shape[1])
    ln_patterns = [
        "blocks.{idx}.ln1.weight",
        "blocks.{idx}.ln0.weight",
        "blocks.{idx}.ln2.weight",
    ]
    for pattern in ln_patterns:
        shape = find_first_shape(model_state, pattern, total_blocks)
        if shape is not None:
            return int(shape[0])
    return None


def infer_rank_dim(model_state, total_blocks, tensor_suffix):
    shape = find_first_shape(model_state, f"blocks.{{idx}}.{tensor_suffix}", total_blocks)
    if shape is not None:
        if len(shape) >= 2:
            return int(shape[1])
        return int(shape[0])
    return None


def infer_intermediate_size(model_state, total_blocks, hidden_size):
    shape = find_first_shape(model_state, "blocks.{idx}.ffn.key.weight", total_blocks)
    if shape is None or len(shape) != 2:
        return None
    dim0, dim1 = int(shape[0]), int(shape[1])
    if hidden_size is not None:
        if dim0 == hidden_size and dim1 != hidden_size:
            return dim1
        if dim1 == hidden_size and dim0 != hidden_size:
            return dim0
    return max(dim0, dim1)


def infer_attention_dims(model_state, total_blocks):
    shape = find_first_shape(model_state, "blocks.{idx}.att.r_k", total_blocks)
    if shape is None or len(shape) != 2:
        return None, None
    return int(shape[0]), int(shape[1])


def update_config_file(config_path, model_state, total_blocks):
    if not config_path:
        return
    if not os.path.exists(config_path):
        print(f"\n警告: 未找到 config 文件 {config_path}，跳过自动更新")
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as exc:
        print(f"\n警告: 读取 config 失败，原因: {exc}")
        return

    hidden_size = infer_hidden_size(model_state, total_blocks)
    num_heads, head_dim = infer_attention_dims(model_state, total_blocks)
    intermediate_size = infer_intermediate_size(model_state, total_blocks, hidden_size)

    decay_low_rank_dim = infer_rank_dim(model_state, total_blocks, "att.w1")
    a_low_rank_dim = infer_rank_dim(model_state, total_blocks, "att.a1")
    v_low_rank_dim = infer_rank_dim(model_state, total_blocks, "att.v1")
    gate_low_rank_dim = infer_rank_dim(model_state, total_blocks, "att.g1")

    updates = {
        "num_hidden_layers": total_blocks,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "decay_low_rank_dim": decay_low_rank_dim,
        "a_low_rank_dim": a_low_rank_dim,
        "v_low_rank_dim": v_low_rank_dim,
        "gate_low_rank_dim": gate_low_rank_dim,
    }

    if num_heads is not None:
        updates.setdefault("num_heads", num_heads)
        config["num_heads"] = num_heads

    for field, value in updates.items():
        if field == "num_heads":
            continue
        if value is not None:
            config[field] = value
        else:
            print(f"警告: 无法从模型中推断 {field}，请手动检查")

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n已更新 config 文件，关键字段如下：")
    for field in ["num_hidden_layers", "hidden_size", "head_dim", "num_heads",
                  "intermediate_size", "decay_low_rank_dim", "a_low_rank_dim",
                  "v_low_rank_dim", "gate_low_rank_dim"]:
        if field in config:
            print(f"  {field}: {config[field]}")


update_config_file(CONFIG_PATH, model, num_blocks)

# 定义全局映射
global_mappings = {
    "emb.weight": "model.embeddings.weight",
    "head.weight": "lm_head.weight",
    "ln_out.weight": "model.norm.weight",
    "ln_out.bias": "model.norm.bias",
}

# 处理全局参数
for old_key, new_key in global_mappings.items():
    if old_key in model:
        new_model[new_key] = model[old_key].clone().contiguous()
        print(f"映射: {old_key} -> {new_key}")
    else:
        print(f"警告: 全局权重 {old_key} 在源模型中未找到")

# 处理每一层
for block_idx in range(num_blocks):
    block_prefix = f"blocks.{block_idx}"
    new_block_prefix = f"model.layers.{block_idx}"
    
    # 定义层归一化映射
    ln_mappings = {}
    # 仅第 0 层有 ln0
    if block_idx == 0:
        ln_mappings.update({
            f"{block_prefix}.ln0.weight": f"{new_block_prefix}.pre_norm.weight",
            f"{block_prefix}.ln0.bias": f"{new_block_prefix}.pre_norm.bias",
        })
    # 所有层有 ln1 和 ln2
    ln_mappings.update({
        f"{block_prefix}.ln1.weight": f"{new_block_prefix}.attn_norm.weight",
        f"{block_prefix}.ln1.bias": f"{new_block_prefix}.attn_norm.bias",
        f"{block_prefix}.ln2.weight": f"{new_block_prefix}.ffn_norm.weight",
        f"{block_prefix}.ln2.bias": f"{new_block_prefix}.ffn_norm.bias",
    })
    
    # 应用层归一化映射
    for old_key, new_key in ln_mappings.items():
        if old_key in model:
            new_model[new_key] = model[old_key].clone().contiguous()
            print(f"映射: {old_key} -> {new_key}")
        else:
            print(f"警告: {old_key} 在源模型中未找到")

    # 处理注意力层权重
    att_prefix = f"{block_prefix}.att"
    new_att_prefix = f"{new_block_prefix}.attn"
    
    att_mappings = {
        f"{att_prefix}.key.weight": f"{new_att_prefix}.k_proj.weight",
        f"{att_prefix}.receptance.weight": f"{new_att_prefix}.r_proj.weight",
        f"{att_prefix}.value.weight": f"{new_att_prefix}.v_proj.weight",
        f"{att_prefix}.output.weight": f"{new_att_prefix}.o_proj.weight",
        f"{att_prefix}.ln_x.weight": f"{new_att_prefix}.g_norm.weight",
        f"{att_prefix}.ln_x.bias": f"{new_att_prefix}.g_norm.bias",
        f"{att_prefix}.k_a": f"{new_att_prefix}.k_a",
        f"{att_prefix}.k_k": f"{new_att_prefix}.k_k",
        f"{att_prefix}.r_k": f"{new_att_prefix}.r_k",
    }
    
    for old_key, new_key in att_mappings.items():
        if old_key in model:
            if old_key.endswith("k_a") or old_key.endswith("k_k"):
                new_model[new_key] = model[old_key].view(-1).clone().contiguous()
            else:
                new_model[new_key] = model[old_key].clone().contiguous()
            print(f"映射: {old_key} -> {new_key}")
        else:
            print(f"警告: {old_key} 在源模型中未找到")

    # 合并 x_* 系列参数为 x_x
    x_keys_ordered = [
        f"{att_prefix}.x_r",
        f"{att_prefix}.x_w",
        f"{att_prefix}.x_k",
        f"{att_prefix}.x_v",
        f"{att_prefix}.x_a",
        f"{att_prefix}.x_g",
    ]
    
    all_x_tensors = []
    for x_key in x_keys_ordered:
        if x_key in model:
            all_x_tensors.append(model[x_key].view(-1).clone().contiguous())
            print(f"使用 {x_key} 构建 x_x")
        else:
            print(f"警告: {x_key} 在源模型中未找到")
    
    if len(all_x_tensors) == 6:
        new_model[f"{new_att_prefix}.x_x"] = torch.stack(all_x_tensors).contiguous()
        print(f"创建合并张量 {new_att_prefix}.x_x，形状为 {new_model[f'{new_att_prefix}.x_x'].shape}")
    else:
        print(f"警告: 无法为 {new_att_prefix}.x_x 创建合并张量，仅找到 {len(all_x_tensors)} 个 x_* 张量")

    # 处理 LoRA 相关权重（修正映射并转置权重）
    lora_mappings = {
        f"{att_prefix}.a0": f"{new_att_prefix}.a_lora.lora.2.bias",
        f"{att_prefix}.a1": f"{new_att_prefix}.a_lora.lora.0.weight",
        f"{att_prefix}.a2": f"{new_att_prefix}.a_lora.lora.2.weight",
        f"{att_prefix}.g1": f"{new_att_prefix}.g_lora.lora.0.weight",
        f"{att_prefix}.g2": f"{new_att_prefix}.g_lora.lora.2.weight",
        f"{att_prefix}.w0": f"{new_att_prefix}.w_lora.lora.2.bias",
        f"{att_prefix}.w1": f"{new_att_prefix}.w_lora.lora.0.weight",
        f"{att_prefix}.w2": f"{new_att_prefix}.w_lora.lora.2.weight",
        f"{att_prefix}.v0": f"{new_att_prefix}.v_lora.lora.2.bias",
        f"{att_prefix}.v1": f"{new_att_prefix}.v_lora.lora.0.weight",
        f"{att_prefix}.v2": f"{new_att_prefix}.v_lora.lora.2.weight",
    }
    
    for old_key, new_key in lora_mappings.items():
        if old_key in model:
            if old_key.endswith("a0") or old_key.endswith("w0") or old_key.endswith("v0"):
                new_model[new_key] = model[old_key].view(-1).clone().contiguous()
            else:
                # 对权重进行转置以匹配维度
                new_model[new_key] = model[old_key].transpose(-1, -2).clone().contiguous()
            print(f"映射: {old_key} -> {new_key}")
        # 注意：v0, v1, v2 仅在第 1–11 层存在
        elif not (old_key.endswith("v0") or old_key.endswith("v1") or old_key.endswith("v2")) or block_idx > 0:
            print(f"警告: {old_key} 在源模型中未找到")

    # 处理 FFN 层权重
    ffn_prefix = f"{block_prefix}.ffn"
    new_ffn_prefix = f"{new_block_prefix}.ffn"
    
    ffn_mappings = {
        f"{ffn_prefix}.key.weight": f"{new_ffn_prefix}.key.weight",
        f"{ffn_prefix}.value.weight": f"{new_ffn_prefix}.value.weight",
        f"{ffn_prefix}.x_k": f"{new_ffn_prefix}.x_k",
    }
    
    for old_key, new_key in ffn_mappings.items():
        if old_key in model:
            if old_key.endswith("x_k"):
                new_model[new_key] = model[old_key].view(-1).clone().contiguous()
            else:
                new_model[new_key] = model[old_key].clone().contiguous()
            print(f"映射: {old_key} -> {new_key}")
        else:
            print(f"警告: {old_key} 在源模型中未找到")

# 打印转换后的模型权重键
print("\n=== 转换后模型权重键 ===")
new_keys_list = sorted(list(new_model.keys()))
print(f"转换后模型中共有 {len(new_keys_list)} 个权重键")
print("\n转换后权重键列表：")
print(new_keys_list)

# 打印转换后权重形状
print("\n=== 转换后模型权重形状 ===")
for key in new_keys_list:
    shape = new_model[key].shape
    print(f"'{key}': {shape}")

# 检查缺失的键
expected_keys = [
    "lm_head.weight",
    "model.embeddings.weight",
    "model.norm.weight",
    "model.norm.bias"
]

for i in range(num_blocks):
    block_prefix = f"model.layers.{i}"
    layer_keys = [
        f"{block_prefix}.attn_norm.weight",
        f"{block_prefix}.attn_norm.bias",
        f"{block_prefix}.attn.k_proj.weight",
        f"{block_prefix}.attn.r_proj.weight",
        f"{block_prefix}.attn.v_proj.weight",
        f"{block_prefix}.attn.o_proj.weight",
        f"{block_prefix}.attn.x_x",
        f"{block_prefix}.attn.k_a",
        f"{block_prefix}.attn.k_k",
        f"{block_prefix}.attn.r_k",
        f"{block_prefix}.attn.g_norm.weight",
        f"{block_prefix}.attn.g_norm.bias",
        f"{block_prefix}.ffn_norm.weight",
        f"{block_prefix}.ffn_norm.bias",
        f"{block_prefix}.ffn.key.weight",
        f"{block_prefix}.ffn.value.weight",
        f"{block_prefix}.ffn.x_k",
        f"{block_prefix}.attn.a_lora.lora.0.weight",
        f"{block_prefix}.attn.a_lora.lora.2.weight",
        f"{block_prefix}.attn.a_lora.lora.2.bias",
        f"{block_prefix}.attn.g_lora.lora.0.weight",
        f"{block_prefix}.attn.g_lora.lora.2.weight",
        f"{block_prefix}.attn.w_lora.lora.0.weight",
        f"{block_prefix}.attn.w_lora.lora.2.weight",
        f"{block_prefix}.attn.w_lora.lora.2.bias",
    ]
    # 仅第 0 层有 pre_norm
    if i == 0:
        layer_keys.extend([
            f"{block_prefix}.pre_norm.weight",
            f"{block_prefix}.pre_norm.bias",
        ])
    # 第 1–11 层有 v_lora
    if i > 0:
        layer_keys.extend([
            f"{block_prefix}.attn.v_lora.lora.0.weight",
            f"{block_prefix}.attn.v_lora.lora.2.weight",
            f"{block_prefix}.attn.v_lora.lora.2.bias",
        ])
    expected_keys.extend(layer_keys)

missing_keys = [key for key in expected_keys if key not in new_keys_list]
if missing_keys:
    print("\n警告: 以下预期键在转换后的模型中缺失:")
    for key in missing_keys:
        print(f"  - {key}")

# 在保存前将所有权重转换为float32
for key in new_model:
    if torch.is_tensor(new_model[key]) and new_model[key].dtype != torch.float32:
        new_model[key] = new_model[key].to(torch.float32)

# 保存新模型
print("\n正在保存新模型到:", TARGET_MODEL)
save_file(new_model, TARGET_MODEL)
print(f"成功将 {SOURCE_MODEL} 转换为 {TARGET_MODEL}")