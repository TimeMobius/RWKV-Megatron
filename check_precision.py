import torch
import os
import argparse
from collections import defaultdict

def get_readable_size(numel):
    """将参数数量转换为易读格式 (K, M, B)"""
    if numel >= 1e9:
        return f"{numel / 1e9:.2f}B"
    elif numel >= 1e6:
        return f"{numel / 1e6:.2f}M"
    elif numel >= 1e3:
        return f"{numel / 1e3:.2f}K"
    else:
        return str(numel)

def inspect_checkpoint(file_path):
    print(f"\n======== Inspecting: {file_path} ========\n")
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        # map_location='cpu' 防止显存爆炸
        state_dict = torch.load(file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 处理可能的嵌套结构 (例如 Megatron 保存的 dict 里面还有 'model' 或 'state_dict' 键)
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            print("[INFO] Found 'state_dict' key, extracting...")
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            print("[INFO] Found 'model' key, extracting...")
            state_dict = state_dict['model']
    
    # 初始化统计
    dtype_counts = defaultdict(int)
    param_counts = defaultdict(int)
    total_params = 0
    
    # 表头格式
    header = f"{'Param Name':<60} | {'Shape':<25} | {'Dtype':<15}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # 遍历打印
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            dtype_str = str(value.dtype).replace("torch.", "")
            shape_str = str(list(value.shape))
            num_params = value.numel()
            
            # 打印单行信息
            print(f"{key:<60} | {shape_str:<25} | {dtype_str:<15}")
            
            # 统计
            dtype_counts[dtype_str] += 1
            param_counts[dtype_str] += num_params
            total_params += num_params
        else:
            print(f"{key:<60} | {'N/A':<25} | {type(value)}")

    # 打印总结
    print("\n" + "="*30)
    print("       SUMMARY REPORT       ")
    print("="*30)
    print(f"Total Parameters: {get_readable_size(total_params)}")
    print("-" * 30)
    
    for dtype, count in dtype_counts.items():
        n_params = param_counts[dtype]
        percent = (n_params / total_params * 100) if total_params > 0 else 0
        print(f"{dtype:<15}: {count:>4} tensors | {get_readable_size(n_params):>8} params ({percent:.1f}%)")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check precision (dtype) of parameters in a .pth file.")
    parser.add_argument("path", type=str, help="Path to the .pth model file")
    
    args = parser.parse_args()
    inspect_checkpoint(args.path)