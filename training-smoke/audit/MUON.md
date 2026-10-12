# 当前 RWKV7 的 Muon 使用与验证

2026-10-10，独立工作区 `RWKV-Megatron-upstream-test`。结论：补齐依赖并使用下面的兼容配置后，完整原始 3B 已通过单卡 BF16 训练、保存和断点续训。尚未验证本配置在多卡 TP/SP/PP/DP 或 FP8 下的正确性，也没有通过短跑证明 Muon 的长期收敛优于 Adam。

## 环境与配置

当前环境 `train_mobius_py312` 原本没有 Muon 依赖，本轮安装 `emerging-optimizers==0.3.0`，与当前 Megatron 的依赖版本一致。安装时使用 `--no-deps`，已有依赖满足要求，Torch 2.9.1+cu128、Triton 3.5.1、TE 2.19.0 未被替换。

```bash
conda activate train_mobius_py312
python -m pip install emerging-optimizers==0.3.0
```

YAML 顶层配置：

```yaml
optimizer: muon
use_distributed_optimizer: true
use_layer_wise_param_layout: false
muon_momentum: 0.95
muon_nesterov: true
muon_split_qkv: false
muon_num_ns_steps: 5
muon_coefficient_type: quintic
muon_scale_mode: spectral
muon_fp32_matmul_prec: medium
muon_tp_mode: duplicated
muon_extra_scale_factor: 1.0
muon_scalar_optimizer: adam
muon_use_syrk: false
ckpt_format: torch
```

完整配置副本为 [config-rwkv7-3b-muon.yaml](../../configs/config-rwkv7-3b-muon.yaml)，默认只跑 4 步、不保存检查点。启动：

```bash
cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
CUDA_VISIBLE_DEVICES=0 bash training-smoke/launch-3b-reference.sh 3b-muon configs/config-rwkv7-3b-muon.yaml
```

学习率沿用 Adam 参考配置的 `1.56e-5`，仅验证能训练，不代表 Muon 的最佳学习率。Adam 参数继续读取 `adam_beta1`、`adam_beta2`、`adam_eps` 等配置。`use_distributed_optimizer: true` 会被框架转为 `use_layer_wise_distributed_optimizer: true`，启动日志中原字段变成 false 是正常转换。RWKV7 没有合并 QKV，关闭 split-QKV；依赖 0.3.0 不支持当前 TP Muon 的 SYRK 路径，因此关闭 SYRK。

旧 `muon_ns_steps` 应改为 `muon_num_ns_steps`；旧 `muon_matched_adamw_rms` 已不被当前 YAML 入口识别，没有可直接照搬的同名配置。不要用旧字段控制新版 Muon。

续训仍使用 `load` 指向本次 Muon 保存的完整目录，并保留相同优化器配置、模型、数据、调度及并行设置。Adam 的优化器状态不能直接恢复给 Muon；改变优化器要按新的训练实验处理。保存/续训具体示例见 [检查点说明](../../configs/config-rwkv7-3b-reference.md#检查点保存与断点续训)。

## 本轮发现的问题

1. **默认新布局的 `torch` 保存接口缺失。** `use_layer_wise_param_layout: true` 时，Muon 和 Adam 被包在外层 `ChainedOptimizer` 中，训练可运行，但保存报 `AttributeError: 'ChainedOptimizer' object has no attribute 'save_state_dict_to_file'`。当前读取分支同样要求 `load_state_dict_from_file`。最初实际保存测试复现了这个错误。暂用现有兼容模式 `use_layer_wise_param_layout: false`，该模式的完整续训已验证。没有修改 Megatron 子模块；默认新布局后续仍值得修复，包括 Muon 状态与 Adam 分布式状态的分别保存，不能只补一个空接口。此发现不意味着 `torch_dist` 已被验证或必然有同一个故障。
2. **逐元素混合系数被误分给 Muon。** TimeMix 的 `mix_coeff_input` 是把六个逐元素系数向量堆成 `[6, hidden]`，不是线性矩阵权重。新版 Muon 默认根据二维形状筛选参数，会将其作为矩阵处理。本轮在参数构造处设置 `use_muon=False`，让它使用 Adam。LoRA 两个矩阵仍使用 Muon，模型公式和 Adam 更新行为不变。该属性在 DDP 构建之前设置，兼容新版的参数分组方式。

另有一次测试调度失误：Adam 与 Muon 对照被同时放到 GPU 0，导致显存不足；已改为逐个运行，并重新完成成功测试。该失败不计入模型正确性结果，错误日志保留。

## 测试结果

H800 单卡，原始完整 3B，BF16，TP1/PP1/SP关闭，seq128，micro batch1/global batch2，MiniPile，固定 eval 4 样本，L2Wrap 1e-4，FP32 key合并开启。

- 连续 4 步和“2 步保存＋恢复后 2 步”均退出码 0，无 NaN、无跳步。
- 训练 loss：`2.751729 → 2.606818 → 2.765862 → 2.580934`。固定 eval：`2.385848 → 2.432406 → 2.440516 → 2.454535`，最终重复 eval 为 `2.454535`。不同优化器更新后的 loss 不要求与 Adam 一致。
- 连续训练和续训的 loss/梯度范数在日志精度下完全一致。保存的模型 968 个张量、优化器 2258 个张量、独立 Muon 状态文件、调度和 RNG 逐项完全一致。
- eval 样本内容哈希与完整精度 loss 在连续训练、每次 eval 和恢复后完全一致；同权重重复 eval 也一致。
- 实际分配：448 个参数张量使用 Muon，454 个使用 Adam；32 层的混合系数全部使用 Adam。测试检查了 embedding/输出层/LoRA/线性矩阵的分配。
- 64 项相关测试通过，包括混合系数分配与“堆叠 Adam 更新等于六个向量独立 Adam 更新”的检查。
- 原 Adam 参考配置重新跑完 4 步和 5 次 eval，loss/梯度范数与原基准在日志精度下完全一致。

短序列吞吐参考：排除第 1 步启动开销，Muon 平均约 1099 ms/步、233 token/s；同配置 Adam 对照平均约 726 ms/步、353 token/s。Muon 在本次 seq128 短跑中每步耗时增加约 51%，吞吐减少约 34%。额外矩阵处理带来开销；只有 3 个计时样本，不能直接推断长序列或多卡速度，也不能据此判断长期训练收益。

证据： [muon-validation.json](muon-validation.json)、[single-resume-muon-20261010-results.json](single-resume-muon-20261010-results.json)、`muon-*-compatible-batches.json` 和 `muon-routing-*-compatible.json`。成功日志分别为 `training-smoke/logs/single-training-{save,resume,continuous}-muon-20261010.log`，Adam 对照为 `3b-adam-control-20261010.log`，回归测试日志为 `muon-regressions.log`。

本轮生成的测试检查点在完成精确比较后删除，只保留配置、测试脚本、日志和结果。清理记录见 `muon-checkpoint-cleanup.json`；用户检查点不在清理范围。
