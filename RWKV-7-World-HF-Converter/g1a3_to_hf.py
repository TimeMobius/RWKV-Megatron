import torch
import os
import json
import argparse
from safetensors.torch import save_file

# ---------------------------------------------------------
# 设置命令行参数
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='将RWKV-7 PTH模型转换为HF Safetensors格式')
parser.add_argument('--source', type=str, required=True, help='源模型路径 (.pth)')
parser.add_argument('--target', type=str, required=True, help='目标保存路径 (.safetensors)')
parser.add_argument('--config', type=str, default=None, help='可选：配置文件的输出路径 (config.json)')
parser.add_argument('--dtype', type=str, default="fp32", choices=["fp16", "bf16", "fp32"], help='输出权重精度')
args = parser.parse_args()

SOURCE_MODEL = args.source
TARGET_MODEL = args.target
TARGET_CONFIG = args.config

# ---------------------------------------------------------
# 辅助函数：配置生成
# ---------------------------------------------------------
def generate_config(model_state, num_layers):
    # 推断 Hidden Size
    if "emb.weight" in model_state:
        hidden_size = model_state["emb.weight"].shape[1]
        vocab_size = model_state["emb.weight"].shape[0]
    else:
        # 备选方案
        hidden_size = model_state["blocks.0.ln1.weight"].shape[0]
        vocab_size = 65536 # 默认值

    # 推断 FFN Intermediate Size
    # 形状通常是 [intermediate, hidden] 或 [hidden, intermediate]
    # g1a3_2.9b: ffn.key.weight [10240, 2560] -> [intermediate, hidden]
    if "blocks.0.ffn.key.weight" in model_state:
        shape = model_state["blocks.0.ffn.key.weight"].shape
        intermediate_size = shape[0] if shape[1] == hidden_size else shape[1]
    else:
        intermediate_size = hidden_size * 4

    # 推断 Attention Head info
    # g1a3_2.9b: att.r_k shape [40, 64] -> [num_heads, head_dim]
    num_heads = 0
    head_dim = 0
    if "blocks.0.att.r_k" in model_state:
        shape = model_state["blocks.0.att.r_k"].shape
        num_heads = shape[0]
        head_dim = shape[1]

    config = {
        "architectures": ["Rwkv7ForCausalLM"], # 假设HF架构名称
        "model_type": "rwkv7", 
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "layer_norm_epsilon": 1e-5,
        "rescale_layer": 0 # RWKV 默认通常为 0
    }
    return config

# ---------------------------------------------------------
# 主转换逻辑
# ---------------------------------------------------------

# 1. 清理旧文件
if os.path.exists(TARGET_MODEL):
    os.remove(TARGET_MODEL)
    print(f"已删除现有的目标文件: {TARGET_MODEL}")

# 2. 加载模型
print(f"正在加载源模型: {SOURCE_MODEL} ...")
# 使用 mmap 加载以节省内存，map_location 放在 CPU
try:
    state_dict = torch.load(SOURCE_MODEL, mmap=True, map_location='cpu')
except:
    # 如果 mmap 失败（旧版 pytorch），尝试普通加载
    state_dict = torch.load(SOURCE_MODEL, map_location='cpu')

# 3. 检测层数
keys = list(state_dict.keys())
block_keys = [k for k in keys if k.startswith("blocks.")]
if not block_keys:
    raise ValueError("未在模型中找到 'blocks.' 开头的键，请检查源模型格式。")
# 获取最大层号
num_layers = max([int(k.split('.')[1]) for k in block_keys]) + 1
print(f"检测到模型层数: {num_layers}")

# 4. 准备新模型的字典
new_state_dict = {}

print("开始转换权重映射...")

for key in keys:
    tensor = state_dict[key]
    new_key = None

    # --- 全局参数映射 ---
    if key == "emb.weight":
        new_key = "model.embeddings.weight"
    elif key == "ln_out.weight":
        new_key = "model.ln_out.weight"
    elif key == "ln_out.bias":
        new_key = "model.ln_out.bias"
    elif key == "head.weight":
        new_key = "head.weight"
    
    # --- Block 内部参数映射 ---
    elif key.startswith("blocks."):
        parts = key.split('.')
        layer_idx = parts[1] # '0', '1', ...
        module = parts[2]    # 'ln1', 'ln2', 'ln0', 'att', 'ffn'
        param_name = ".".join(parts[3:]) # 'weight', 'bias', 'x_r', 'w1', etc.

        # 构建 HF 格式的前缀: model.blocks.X
        prefix = f"model.blocks.{layer_idx}"

        if module == "ln0":
            # 特例: blocks.0.ln0 -> model.pre_ln (根据你的附件1b5文件)
            new_key = f"model.pre_ln.{param_name}"
        
        elif module == "ln1":
            new_key = f"{prefix}.ln1.{param_name}"
        
        elif module == "ln2":
            new_key = f"{prefix}.ln2.{param_name}"
        
        elif module == "att":
            # att -> attention
            # 参数名完全保持一致 (x_r, w1, a1, receptance.weight 等均直接对应)
            new_key = f"{prefix}.attention.{param_name}"
        
        elif module == "ffn":
            # ffn -> feed_forward
            new_key = f"{prefix}.feed_forward.{param_name}"
    
    # --- 执行转换 ---
    if new_key:
        # 维度检查：附件显示源和目标的维度方向是一致的，不需要转置
        # 源 ffn.key: [10240, 2560], 目标 ffn.key: [8192, 2048] -> 都是 [Intermediate, Hidden]
        # 源 att.w1: [2560, 96], 目标 att.w1: [2048, 96] -> 都是 [Hidden, Rank]
        
        # 数据类型转换
        if args.dtype == "fp32":
            tensor = tensor.float()
        elif args.dtype == "fp16":
            tensor = tensor.half()
        elif args.dtype == "bf16":
            tensor = tensor.bfloat16()
            
        new_state_dict[new_key] = tensor
        # 可选：打印部分日志
        # print(f"Mapped: {key} -> {new_key} | {tensor.shape}")
    else:
        print(f"警告: 发现未处理的键 {key}，将被忽略")

# 5. 生成并保存 Config
if TARGET_CONFIG:
    print(f"正在生成配置文件: {TARGET_CONFIG}")
    config_data = generate_config(state_dict, num_layers)
    with open(TARGET_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)

# 6. 保存 Safetensors
print(f"正在保存模型到 {TARGET_MODEL} (总参数: {len(new_state_dict)}) ...")
save_file(new_state_dict, TARGET_MODEL)
print("转换完成！")