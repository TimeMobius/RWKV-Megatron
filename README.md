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

## Qwen→RWKV 蒸馏分支功能范围

以下清单记录分支 `codex/qwen-rwkv28b-distill` 的当前约定。实现完成后将对应项标记为 `[x]`。

| 功能 | 状态 | 决策与说明 |
|---|---|---|
| 按组件/参数名冻结 | 不做 | 保持当前全参数训练策略。 |
| 更换冻结策略后的恢复流程 | 不做 | 冻结策略不在当前范围内。 |
| 独立 student CE/PPL 监控 | `[x]` 已完成 | 始终统计有效词表上的 student CE，并独立上报 student PPL；CE 只有配置权重非零时才参与训练目标。 |
| 大规模 JSONL 懒加载 | `[x]` 已完成 | 初始化只建立文件/行偏移索引，JSON 解析与 tokenize 延迟到取样时执行。 |
| 递归 JSONL 文件读取 | `[x]` 已完成 | 目录输入递归发现 `.jsonl`，按 split 文件名优先选择并保持稳定排序。 |
| 本地 teacher 后端 | 不做 | 继续使用外部 vLLM teacher 服务。 |
| 直接发送 token ID | `[x]` 已实现；当前网关不支持 | `rwkv_teacher_prompt_mode: token_ids`；文本模式仍为默认，当前 API 对数组 prompt 返回 422。 |
| 多 GPU 蒸馏（TP/PP/SP/DP2） | `[x]` 已实现 | DP2/DP4样本、精度和严格恢复通过；deterministic_mode固定FLA scan，恢复末位问题已定位并修复，见docs/distillation-replay-20261026.md。 |
| 跨 rank teacher 失败处理 | `[x]` 已实现 | PP 末段 TP0 请求，TP 组广播结果/异常；失败由 torchrun 终止全作业，不跳样本。 |
| Teacher pool 与训练一键启动 | `[x]` 已完成 | `tools/launch_distillation.py` 启动 pool，等待就绪后启动训练，并在退出或失败时清理进程组。 |
| 普通 checkpoint 保留最近 N 个并清理旧文件 | `[x]` 已完成 | `rwkv_checkpoint_keep_last: N`，0 为关闭；项目侧回调实现，无需修改 Megatron 子仓库。 |
| Hidden-state 蒸馏 | 不做 | 源项目也没有接入在线训练。 |
| Attention 蒸馏 | 不做 | 源项目也没有接入在线训练。 |
| Qwen→RWKV 权重转换 | 不做 | 使用现有 mapped RWKV checkpoint。 |
| 持久化 teacher 响应数据库 | 不做 | 两个项目目前都不提供此功能。 |

当前蒸馏分支已经完成：在线 vLLM top-k prompt-logprob teacher、`sk-distill` 请求头、固定 seed、请求温度 1.0、有效词表 KL、padding 行排除、重试退避、限流统计、缓存、异步预取、动态并发、teacher pool、JSONL 文本输入、round-trip 检查、checkpoint 恢复和合并后的 W&B 监控。

运行环境固定为 Conda `train_mobius_py312`。Teacher 请求使用 `Authorization: Bearer sk-distill`、`rwkv_teacher_seed: 1234`、请求温度 `1.0` 和 top-100；蒸馏温度由本地 KL 独立处理。开启 `log_timers_to_tensorboard: true` 并配置 TensorBoard/W&B，以便同时记录吞吐和迭代耗时。`throughput` 的单位是 TFLOP/s/GPU，`iteration-time` 的单位是 s/step。

`distill/student_ce_loss` 使用原始 next-token 标签和有效 token mask，在有效词表上计算自然对数 CE（含最后一个可预测标签），独立于 teacher top-k 和 KL 温度；默认 `rwkv_distill_ce_weight: 0.0`，监控不保留反向图。每个优化步的 `distill/student_ppl` 由跨微批次汇总后的 token CE 取 `exp` 得到；验证 PPL 在整个验证集批次汇总后计算。验证/测试使用 `distill/validation_student_ce_loss`、`distill/validation_student_ppl` 和 `distill/test_student_*` 独立命名，避免 Megatron 通用日志将 test 写入带 validation 的同名指标。日志区间中显示的训练 PPL 是这些优化步 PPL 的均值。测试 `tests/test_distill_telemetry.py` 对齐直接 PyTorch CE、有效词表以及 FP32/BF16 纯 KL 梯度，5 项通过。

JSONL 构造阶段只扫描字节建立每行 32 字节的偏移索引，不解析 JSON、不 tokenize；分块索引按访问需要建立，每个窗口保存 32 字节整数偏移。文本/分词结果不常驻内存；长文档的全部可用分块按原顺序保留，超过训练长度的字符块递归二分，不只取文档开头。`target_samples` 已知时 `len()` 不触发 tokenize；未指定样本数时，`len()` 需要建立完整窗口索引。随机访问较后的样本需先建立其前面的窗口索引；多 worker 会分别维护索引，超大数据建议先用较少 worker。递归目录和 `**/*.jsonl` 模式均可用，目录中的 train/validation/test 子目录或文件名用于区分 split；显式 glob 按匹配结果读取。`tests/test_jsonl_dataset.py` 覆盖原始分块算法、重复/随机访问、多 worker 续读和嵌套 split，共 8 项通过。

`rwkv_teacher_prompt_mode` 默认为 `text`，可设为 `token_ids` 直接发送整数 ID 数组；JSONL 的右侧 padding 被移除，binidx 使用完整输入窗口。该设置也参与 pool 握手校验。2026-10-21 实测 `http://192.168.10.121:8000/v1/completions` 的文本请求正常，数组 prompt 返回 HTTP 422 `ExpectedString`；token-ID 模式目前适用于支持该 vLLM 协议的服务，不会在失败时静默退回文本模式。文本窗口包含 S+1 个 token（末 token 为标签）时，teacher 评分后只返回前 S 个位置，保证与 student 的 S 个输入对齐。

在已激活并配置 CUDA 库路径的 `train_mobius_py312` 环境中，一键启动共享 pool 和单卡训练：

```bash
python tools/launch_distillation.py --rwkv-config configs/config-rwkv7-qwen-distill-qwenvocab-2p7b-pool-smoke.yaml
```

Launcher 使用同一份 YAML 配置两个进程，并为训练追加 `--rwkv-teacher-pool`；不会启动 vLLM。训练正常/异常退出、pool 启动失败/超时、训练期间 pool 退出或 SIGINT/SIGTERM 都会清理所属进程组；已占用的 pool 地址会报错。需要自定义训练命令时，放在 `--` 后（如 `-- torchrun --standalone --nproc-per-node=1 pretrain_rwkv7.py`），配置和 pool 参数由 launcher 追加。生命周期测试 `tests/test_distill_launcher.py` 共 6 项通过；实际 2.7B student、minipile JSONL、真实 API、pool/异步、每步两个微批次的 4 步训练与验证已正常结束，零 NaN、零跳步。

`rwkv_checkpoint_keep_last: 2` 保留最近两个已完成的普通 checkpoint；默认 `0` 不清理。以 `latest_checkpointed_iteration.txt` 为完成边界，在下一步开始/训练结束时清理，并在 `pretrain()` 返回后检查最终保存。缺失/无效 tracker 或最新 checkpoint 不完整时不删除；较新的在写目录、release、初始化 step 0、符号链接及非 checkpoint 目录保留。该策略不能与 Megatron 的 `save_retain_interval` 同时开启。测试覆盖删除边界、保留数量及 rank 0 行为；实际小模型同步 torch checkpoint 在 step 1–4 保存后只保留 3、4，从 4 恢复到 5 后只保留 4、5。更改训练总步数后续训时，应保持原 scheduler 配置，或显式 `use_checkpoint_opt_param_scheduler: true` 恢复 checkpoint 中的 scheduler。

单卡功能测试详情见 [蒸馏功能验证记录](docs/distillation-validation-20261021.md)。TP/PP/SP 的分阶段计划、精度阈值、恢复和多卡测试见 [多卡蒸馏记录](docs/distillation-parallel-20261023.md)。普通PP与SP、DP2/DP4、分布式优化器和CPU offload已通过精度及严格恢复。历史[DP2记录](docs/distillation-dp2-20261023.md)中的恢复差异已由deterministic_mode固定FLA scan解决；普通自动调优模式不承诺位一致。连续CP2的组合精度、严格恢复、真实2.7B评测和在线短训通过，范围见[剩余工作记录](docs/distillation-remaining-20261026.md)；虚拟/交错PP仍拒绝，多机未认证。

多卡 launcher 从 YAML 的 DP×TP×PP×CP 计算本机进程数；DP默认1，在 `model_parallel.data_parallel_size` 设置。例如以下配置自动启动8卡DP2×TP2×PP2+SP：

```yaml
model_parallel:
  data_parallel_size: 2
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 2
  sequence_parallel: true
  context_parallel_size: 1
```

CP使用连续切分、真实recurrent状态传递；CP-local序列需按64token切块，SP还需按TP切分。recurrent CP按rank传递状态，尚未承诺加速收益。启动前检查并行度、SP/TP/CP、head/hidden切分、普通PP层数分配、global batch能否按micro batch×DP划分，以及可见GPU数。显式torchrun命令仍可用；如配置已指定DP，进程数必须一致。原4卡TP2+PP2+SP配置仍可这样启动：

```bash
bash training-smoke/parallel-distill-20261023/run.sh python tools/launch_distillation.py \
  --rwkv-config configs/config-qwen-2p7b-distill-tp2-sp-pp2-smoke.yaml
```

教师只在最后一个 PP stage 的 CP0/TP0 请求/异步预取完整上下文，先分发CP目标，再广播给其他 TP ranks；student 词表保持分片，KL/CE 排除全局 padding 行。配置的词表 padding 必须与 native checkpoint 行数一致：当前 248320 行 Qwen 初始化在 TP4 下使用 `make_vocab_size_divisible_by: 128`，TP2 下使用 256。生产在线 smoke 需要外部 Qwen 服务健康；固定教师测试用于验证并行与精度，不能代替真实 teacher 的训练质量评估。2026-10-23 外部服务恢复后，2.7B 在 dense、TP2、TP2+SP、PP2、TP2+PP2+SP、TP4+PP2+SP 六种布局均完成真实 Qwen 在线3步训练及验证/test，零NaN/skip；[W&B验收报告](http://dian.sea-group.org:9098/mobius/rwkv-qwen-distill-2p7b/runs/v4pf5ud7)与逐布局结果见上述多卡记录。

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

这是 4 步短跑模板，固定学习率为 `1.56e-5`，默认不写大检查点。正式训练先另存一份，再调整序列长度、batch、训练步数、学习率调度和 `save` 输出目录；续训使用 `load` 指向完整训练检查点。当前格式由 `ckpt_format` 决定，旧的 `use_dist_ckpt` / `dist_ckpt_format` 已不生效。保存、续训及自动识别格式的 YAML 示例见 [检查点说明](configs/config-rwkv7-3b-reference.md#检查点保存与断点续训)。

FP8 默认关闭。实验时在 YAML 中设置 `language_model.fp8: hybrid`、`language_model.fp8_recipe: tensorwise` 和 `transformer_impl: transformer_engine`；`rwkv_fp8_scope: ffn` 限定为 FFN 的两个大矩阵乘法。当前测试仍建议正式训练使用 BF16，精度和速度记录见 [FP8.md](training-smoke/audit/FP8.md)。配置解析与实际短跑记录见 [参数验证](training-smoke/audit/3b-reference-config-validation.json) 和 [训练验证](training-smoke/audit/3b-reference-training-validation.json)。

Muon 依赖 `emerging-optimizers`，当前测试环境安装的是 `0.3.0`。在 YAML 顶层配置：

```yaml
optimizer: muon
use_distributed_optimizer: true
use_layer_wise_param_layout: false
muon_momentum: 0.95
muon_nesterov: true
muon_split_qkv: false
muon_num_ns_steps: 5
muon_scale_mode: spectral
muon_tp_mode: duplicated
muon_scalar_optimizer: adam
```

完整测试副本见 [config-rwkv7-3b-muon.yaml](configs/config-rwkv7-3b-muon.yaml)。矩阵权重和 LoRA 矩阵使用 Muon；embedding、输出层、归一化及逐元素混合系数使用 Adam。`use_distributed_optimizer: true` 会由新版 Megatron 自动转为 Muon 的层级分布式优化器路径。当前默认新布局保存 `torch` 检查点会报 `ChainedOptimizer` 缺少保存接口，暂用 `use_layer_wise_param_layout: false` 兼容模式。旧的 `muon_ns_steps` 应改为 `muon_num_ns_steps`；`muon_matched_adamw_rms` 已不被当前解析器识别，不能照搬旧配置。依赖 `0.3.0` 时保持 `muon_use_syrk: false`。

```bash
CUDA_VISIBLE_DEVICES=0 bash training-smoke/launch-3b-reference.sh 3b-muon configs/config-rwkv7-3b-muon.yaml
```

模板学习率仅用于短跑验证，Muon 的正式训练学习率需要单独调试。Adam 检查点不能直接恢复为 Muon 优化器状态；切换优化器属于新的训练实验。单卡测试与续训记录见 [MUON.md](training-smoke/audit/MUON.md)。

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

RWKV7 在 `sequence_parallel: true` 时，`v0` 自动只保留本 TP 卡的通道，并以两份数据（隐藏状态＋本地 `v0`）跨普通 PP 传递。TP2/TP4 的完整消息体分别减少约 33%/60%。模型参数和检查点张量形状不变；主仓库与 Megatron 子模块的通信形状改动需要一起使用。完整 3B 前后向及两种 8 卡组合的固定 eval/断点续训已验证，细节见 [v0 分片报告](training-smoke/audit/V0-SHARD-20261012.md)。目前仍使用非交错 PP、CP1；交错 PP、FP8 和 Muon 多卡另行验证。

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
