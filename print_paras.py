import torch
import megatron
# 加载模型权重
model_path = '/mnt/data/Outs/rwkv7-megatron/torch/iter_0000050/mp_rank_00/model_optim_rng.pt'
model_weights = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
total_params = 0

for key, value in model_weights.items():
    total_params += value.numel()
    # print(key, value.shape)
    print(f"Parameter: {key}, value: {value}, Shape: {value.shape}, Dtype: {value.dtype}")

print(f"Total parameters: {total_params}")
