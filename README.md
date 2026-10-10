# RWKV-Megatron

RWKV7 的默认 `ln0`、各层 `ln1/ln2` 与 `ln_out` 统一使用 Transformer Engine >=1.9.0 的 LayerNorm；TE 缺失或版本不足时统一回退 Apex FusedLayerNorm（需要 Apex CUDA 扩展）。该选择独立于 `transformer_impl` 的 Linear 后端。GroupNorm 保持原公式；此变更不代表已完成 FP8 全模型训练认证。

RWKV7 的 key 归一化、更新和 a/b 生成统一使用 FP32 中间计算，最终结果保留投影输出的原类型。CUDA 默认通过 `torch.compile` 合并计算；编译不可用时提示并回退普通 Torch FP32。YAML 设置 `rwkv_key_fusion: false` 可固定选择兼容路径；未设置时仍支持环境变量 `RWKV7_KEY_FUSION=0`。首次运行有编译开销，断点续训应保持相同后端。测试记录见 [KEY-PRECISION-INTEGRATION.md](training-smoke/audit/KEY-PRECISION-INTEGRATION.md)。

RWKV7 可以在原 CE 后端上增加官方式 L2Wrap 梯度；默认系数为 `0`（关闭），官方系数为 `0.0001`。在 `--rwkv-config` 指定的 YAML 顶层配置：

```yaml
rwkv_l2wrap_coefficient: 0.0001
rwkv_key_fusion: true
```

L2Wrap 不改变显示的 CE loss；只在训练后向中，对每个有效 token 的全词表最大 logit 增加梯度。它跟随原 mask、token 数归一化和梯度缩放；被 mask 的 token 不增加正则梯度。YAML 中明确设置的 key 选项优先于环境变量，命令行 `--rwkv-l2wrap-coefficient`、`--rwkv-key-fusion` / `--no-rwkv-key-fusion` 可覆盖 YAML。测试范围与结果见 [L2WRAP.md](training-smoke/audit/L2WRAP.md)。

验证默认继续读取下一组样本。YAML 顶层设置 `fixed_eval_samples: true` 后，每次 eval 都从验证集开头读取同一组；断点恢复后同样从开头读取。组的大小由 `eval_iters` 和验证 batch size 决定，比较时应保持数据、seed、序列长度及验证批次配置相同。设为 `false` 或不设置则保留原行为。支持内置 `single`/`cyclic` DataLoader；外部自定义迭代器不支持此开关。测试见 [FIXED-EVAL.md](training-smoke/audit/FIXED-EVAL.md)。

<div align="center">

<img src="/assets/logo.png" width="300px">

**基于 Megatron-Core 构建的 RWKV 架构大模型高效训练框架**

[![license](https://img.shields.io/github/license/BlinkDL/RWKV-LM)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

简体中文 | [English](README_EN.md)
</div>

## 📖 项目简介

RWKV-Megatron 是一个面向大规模分布式训练场景的 RWKV 模型训练框架，基于 NVIDIA Megatron-Core 构建，并结合 RWKV 架构在线性复杂度建模方面的特性，对训练入口、模型实现、参数配置、权重初始化与检查点转换流程进行了系统化集成。

本仓库的目标并不仅仅是“运行 RWKV”，而是提供一套可用于中大规模实验与工程训练的统一训练栈，使 RWKV 模型能够直接复用 Megatron-Core 在并行训练、混合精度、内存管理与分布式优化器方面的成熟能力。

**RWKV 架构优势**：
- 同时具备 RNN 式高效推理与 Transformer 式并行训练能力
- 线性时间复杂度建模路径，更适合长序列训练与部署场景
- 在长上下文建模、内存效率与推理吞吐之间具有良好的工程平衡

**Megatron-Core 框架优势**：
- 提供模块化的大模型训练基础设施与分布式并行抽象
- 具备成熟的张量并行、流水线并行、序列并行与数据并行能力
- 内置混合精度、激活重计算、分布式优化器与检查点机制

## 🚀 核心特性

### 模型架构
- **原生支持 RWKV6 / RWKV7**：仓库提供独立的 RWKV6 与 RWKV7 训练入口与模型实现
- **Megatron-Core 适配实现**：RWKV 模块按 Megatron-Core 训练范式接入，便于复用其并行与调度能力
- **模块化参数配置**：采用 YAML 配置组织模型结构、训练超参数、并行策略与日志行为
- **多来源初始化能力**：支持从 `rwkv-lm` 权重与 `rwkv-huggingface` 权重初始化 RWKV7 模型

### 分布式训练
- **多维并行训练能力**：
  - **张量并行** (Tensor Parallelism)
  - **流水线并行** (Pipeline Parallelism)
  - **序列并行** (Sequence Parallelism)
  - **数据并行** (Data Parallelism)
- **YAML 驱动的训练组织方式**：便于在不同硬件规模下快速调整模型与并行配置

### 性能优化
- **混合精度训练**：支持 FP16 与 BF16 训练模式
- **激活重计算**：支持通过重计算降低训练显存占用
- **分布式优化器**：支持 Megatron 分布式优化器路径
- **Muon 优化器支持**：底层优化器栈已接入 Muon，可用于对比 Adam 系列优化器的训练行为
- **通信与计算优化选项**：支持梯度累积融合、并行通信重叠等 Megatron-Core 常见优化开关

### 互操作性
- **检查点与权重转换工具链**：
  - 支持从 Megatron 分布式检查点转换为 RWKV 原始权重格式

## 💡 使用场景

- **大规模预训练**：面向多卡、多节点场景组织 RWKV 预训练任务
- **结构实验与训练对比**：用于比较不同并行策略、精度模式与优化器配置
- **领域模型训练与继续预训练**：基于现有 RWKV 权重开展领域适配训练
- **权重格式转换与工程集成**：为后续推理框架或 HuggingFace 生态适配提供中间转换能力

## 🔧 快速开始

### 安装与环境配置

建议优先使用仓库内维护的 Conda 环境文件：

```bash
# 克隆仓库及子模块
git clone --recursive https://github.com/shiroko98/RWKV_Megatron.git
cd RWKV_Megatron

# 创建并激活环境
conda env create -f environment.yml
conda activate train_rwkv_nvidia_megatron_rwkv7
```

当前仓库根目录中的 `megatron` 与 `fla` 已分别指向：

```bash
megatron -> 3rdparty/Megatron-LM-0.11.0/megatron
fla      -> 3rdparty/flash-linear-attention-260115/fla
```

若使用自定义环境，请确保至少具备以下依赖栈：
- Python 3.10+
- PyTorch 2.2+
- CUDA / NCCL
- FlashAttention
- DeepSpeed
- Transformers

### 验证环境

可通过以下脚本检查 PyTorch 与 CUDA 环境：

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## 📘 使用方法

安装完成后，典型工作流程包括：
1. 数据预处理
2. 预训练或继续预训练
3. 权重导出与格式转换

### 一、数据预处理

训练数据采用宽松的 `jsonl` 格式，每行至少包含文本字段，例如：

```json
{"text": "The quick brown fox"}
{"text": "jumps over the lazy dog"}
```

可以使用仓库中的 `preprocess_data.py` 将文本数据转换为 Megatron 可直接读取的 `bin / idx` 格式：

```bash
python preprocess_data.py \
  --input /path/to/data.jsonl \
  --output-prefix /path/to/output/train \
  --json-keys text \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /path/to/tokenizer \
  --append-eod
```

处理完成后，通常会生成如下文件：
- `train_text_document.bin`
- `train_text_document.idx`

在训练配置中，`data_path` 应填写为该文件前缀，而非具体扩展名。

### 二、训练 RWKV 模型

#### 配置 YAML 示例

当前 RWKV7 3B 优先参考 [config-rwkv7-3b-reference.yaml](configs/config-rwkv7-3b-reference.yaml)，详细说明见 [配置说明](configs/config-rwkv7-3b-reference.md)。它已在 H800 单卡、Python 3.12.14 / Torch 2.9.1+cu128 / TE 2.19.0 环境中完成完整 3B 的 4 步训练和 5 次 eval，零 NaN、零跳步。116 个配置项均被当前解析器识别或明确转换。

其他配置见 [configs](./configs)。`example_config.yaml` 是历史模板，仍包含旧字段，缺少当前 RWKV7 的四个 LoRA 尺寸，不能直接作为新版 3B 的运行配置。

#### 已验证的 3B 参考配置

以下为关键参数，运行时使用上面链接的完整 YAML：

```yaml
language_model:
  num_layers: 32
  hidden_size: 2560
  num_attention_heads: 40
  kv_channels: 64
  ffn_hidden_size: 10240
  D_AAA_LORA: 96
  D_DECAY_LORA: 96
  D_GATE_LORA: 320
  D_MV_LORA: 64
  activation_func: squaredrelu
  fp8: false

model_parallel:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  context_parallel_size: 1
  sequence_parallel: false
  bf16: true

transformer_impl: local
rwkv_key_fusion: true
rwkv_l2wrap_coefficient: 0.0001
fixed_eval_samples: true
rwkv_fp8_scope: ffn
optimizer: adam
lr: 1.56e-05
lr_decay_style: constant
micro_batch_size: 1
global_batch_size: 2
seq_length: 128
train_iters: 4
eval_iters: 2
eval_interval: 1
save: null
```

完整配置使用原始 3B `.pth` 权重、MiniPile 的 `.bin/.idx` 前缀和 XiaoKe tokenizer。词表有 65,534 个条目，按最高 ID 加一并补齐后的模型行数为 65,536，由 tokenizer 自动计算。每次 eval 使用相同的 4 个样本，断点恢复后也使用同一组。

当前本地工作区可以直接运行配套脚本：

```bash
CUDA_VISIBLE_DEVICES=0 bash training-smoke/launch-3b-reference.sh
```

脚本激活 `train_mobius_py312`、配置 CUDA 库路径，并调用 `pretrain_rwkv7.py --rwkv-config configs/config-rwkv7-3b-reference.yaml`。配置和脚本中的绝对路径对应当前本地环境；换机器或工作区时需修改环境、仓库、模型、数据及 tokenizer 路径。

这是 4 步短跑模板，固定学习率为 `1.56e-5`，默认不写大检查点。正式训练先另存一份，再调整序列长度、batch、训练步数、学习率调度和 `save` 输出目录；续训使用 `load` 指向完整训练检查点。当前格式由 `ckpt_format` 决定，旧的 `use_dist_ckpt` / `dist_ckpt_format` 已不生效。

FP8 默认关闭。实验时在 YAML 中设置 `language_model.fp8: hybrid`、`language_model.fp8_recipe: tensorwise` 和 `transformer_impl: transformer_engine`；`rwkv_fp8_scope: ffn` 限定为 FFN 的两个大矩阵乘法。当前测试仍建议正式训练使用 BF16，精度和速度记录见 [FP8.md](training-smoke/audit/FP8.md)。配置解析与实际短跑记录见 [参数验证](training-smoke/audit/3b-reference-config-validation.json) 和 [训练验证](training-smoke/audit/3b-reference-training-validation.json)。

如需启用 Muon，可在优化器配置中切换为：

```yaml
optimizer: muon
```

Muon 相关超参数沿用 Megatron 参数命名方式，例如 `muon_matched_adamw_rms`、`muon_momentum` 与 `muon_ns_steps`。

#### 分布式训练配置

可以根据硬件资源组织不同的并行策略：

```python
# 8-GPU 张量并行
--tensor-model-parallel-size 8 --pipeline-model-parallel-size 1

# 2-GPU 张量并行 × 4-GPU 流水线并行
--tensor-model-parallel-size 2 --pipeline-model-parallel-size 4

# 启用序列并行
--sequence-parallel --tensor-model-parallel-size 8
```

#### 编写启动脚本

**RWKV6 模型**

```bash
#!/bin/bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=24
export CUDA_DEVICE_MAX_CONNECTIONS=1

conda activate train_rwkv_nvidia_megatron_rwkv7
cd /path/to/RWKV_Megatron

torchrun --master_port 51281 --nproc-per-node 8 --nnodes 1 \
  pretrain_rwkv6.py --yaml-cfg /path/to/config_rwkv6.yaml | tee output.log
```

**RWKV7 模型**

```bash
#!/bin/bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=24
export CUDA_DEVICE_MAX_CONNECTIONS=1

conda activate train_rwkv_nvidia_megatron_rwkv7
cd /path/to/RWKV_Megatron

torchrun --master_port 51281 --nproc-per-node 8 --nnodes 1 \
  pretrain_rwkv7.py --rwkv-config /path/to/config_rwkv7.yaml | tee output.log
```

仓库中亦提供了一个简单示例脚本：

- `example_pretain_rwkv6.sh`

### 三、模型权重转换

#### 1. Megatron 分布式检查点转换为 RWKV 原始权重

```bash
python rwkv_conversion.py \
  --load-dir /path/to/model \
  --save-dir /path/to/save
```

默认读取根目录的 `latest_checkpointed_iteration.txt`。指定旧迭代时可传
`--iteration 121500`，此时不读取 tracker；也可直接传入
`--load-dir /path/to/model/iter_0121500/`。同时提供目录和迭代时，两者必须一致。
输出仍为 `/path/to/save/iter_0121500/rwkv_state_dict.pth`。

`--norm-offset` 默认根据 checkpoint 的 `layernorm_zero_centered_gamma` 自动选择
`+1` 或 `0`，仅作用于 `ln0`、`ln1`、`ln2` 和 `ln_out` 的 gamma；显式传入
`--norm-offset 0` 可关闭偏移。GroupNorm 和所有 bias 保持不变。

脚本在保存前严格检查全部 RWKV 参数的 key、shape 和有限值，以及 TP shard
的一致性；失败不会覆盖已有模型或更新输出 tracker。成功时通过临时文件原子替换
模型，再更新 tracker。目前支持均匀 PP 分层，不支持虚拟流水线或非均匀分层。
注意力头 shape 按训练配置导出，不会自动转换为另一种头分组。

转换器单元测试和 TP/PP 回归测试：

```bash
python -m unittest discover -s tests -p 'test_rwkv_conversion.py' -v
```

#### 2. `pth -> HuggingFace` 相关格式转换

如需进行 `pth -> HuggingFace` 相关格式转换，可访问独立仓库：

- [shiroko98/RWKV-7-World-HF-Converter](https://github.com/shiroko98/RWKV-7-World-HF-Converter)

## 🤝 如何贡献

我们欢迎各种形式的贡献，包括但不限于：

1. 报告问题与提交功能请求
2. 改进模型实现、训练配置或工具脚本
3. 完善文档与使用说明
4. 分享训练经验、优化策略与实验结论

## 📚 引用与致谢

如果您认为本仓库对您的研究或工程工作有帮助，欢迎引用：

```bibtex
@software{Liu2025RWKV_Megatron,
  title  = {RWKV_Megatron: A high-performance RWKV model training framework based on NVIDIA Megatron-Core},
  author = {Liu, Mingda and Zhang, Guozhen},
  url    = {https://github.com/TimeMobius/RWKV-Megatron},
  month  = May,
  year   = {2025}
}
```

本项目的实现与工程实践建立在以下开源项目之上：

- [Megatron-LM / Megatron-Core](https://github.com/NVIDIA/Megatron-LM)
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

## 📄 许可证

本项目采用 [LICENSE](LICENSE) 许可证开源。
