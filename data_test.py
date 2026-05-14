import os
import math
import sys

# --- 您需要修改的部分 ---
# 1. 将 Megatron-LM 项目的根目录添加到 Python 路径中
#    请确保这个路径是正确的，以便脚本能找到 megatron 库
megatron_project_path = '/mnt/data/Codes/RWKV/RWKV-Megatron-dev/megatron' 
sys.path.append(megatron_project_path)

# 2. 您的数据文件前缀列表 (从您的 YAML 配置文件中复制)
data_prefixes = [
    '/mnt/data/Datasets/sft_data/sft_data_text_document',
    '/mnt/data/Datasets/SFT_Data_Origin/SFT_Data_Origin_text_document'
]

# 3. 全局批次大小 (从您的配置文件中获取)
global_batch_size = 128

# 4. 训练集划分比例 (通常 Megatron 在预处理时已完成划分，这里假设所有数据都用于训练)
#    如果您的数据是混合的，并且在加载时有特殊处理，您可能需要调整这里的逻辑。
#    但对于预处理好的数据，通常直接计算总数。
# --- 结束 ---

try:
    # 导入 Megatron 自己的数据加载工具
    from megatron.data.indexed_dataset import MMapIndexedDataset
    print("成功导入 Megatron 的 MMapIndexedDataset 工具。")
except ImportError:
    print(f"错误：无法从 '{megatron_project_path}' 导入 Megatron 库。")
    print("请确认上面的 'megatron_project_path' 变量设置是否正确。")
    sys.exit(1)

total_samples = 0
print("\n正在使用 Megatron 工具读取数据集大小...")

for prefix in data_prefixes:
    # 检查对应的 .bin 和 .idx 文件是否存在
    if os.path.exists(prefix + '.bin') and os.path.exists(prefix + '.idx'):
        try:
            # 使用 MMapIndexedDataset 来加载数据
            dataset = MMapIndexedDataset(prefix)
            num_samples = len(dataset)
            total_samples += num_samples
            print(f"成功读取: {os.path.basename(prefix)}，包含 {num_samples} 个样本。")
        except Exception as e:
            print(f"读取文件前缀 {prefix} 时出错: {e}")
    else:
        print(f"警告：找不到文件对 {prefix}.bin / .idx，已跳过。")

if total_samples == 0:
    print("\n错误：未能成功加载任何数据集。请检查您的文件前缀是否正确。")
else:
    # 计算一个 epoch 能提供的总批次数量
    total_batches_per_epoch = math.floor(total_samples / global_batch_size)

    print("\n" + "="*30)
    print("      数 据 集 诊 断 报 告 (Megatron 格式)")
    print("="*30)
    print(f"所有文件中的样本总数: {total_samples}")
    print(f"全局批次大小 (global_batch_size): {global_batch_size}")
    print(f"每个 Epoch 的最大迭代步数: {total_batches_per_epoch}")
    print("="*30)
    print("\n请将您配置文件中的 `train_iters` 与上面的'最大迭代步数'进行比较。")