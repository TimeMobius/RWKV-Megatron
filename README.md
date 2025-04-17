
## RWKV-Megatron Overview

此项目是使用Megatron-Core用于RWKV架构大模型训练的脚本和针对Megatron的patch，该资源库由两个基本组件组成：Megatron-LM 和 Megatron-Core。Megatron-LM 是一个面向研究的框架，利用 Megatron-Core 进行大型语言模型（LLM）训练。Megatron-Core 是一个 GPU 优化训练技术库，具有正式的产品支持，包括版本化 API 和定期发布。

Megatron-Core 是一个基于 PyTorch 的开源库，包含 GPU 优化技术和尖端的系统级优化。它将这些技术抽象为可组合的模块化 API，使开发人员和模型研究人员能够在英伟达™（NVIDIA®）加速计算基础设施上灵活地大规模训练自定义变换器。该库兼容所有英伟达™（NVIDIA®）Tensor Core GPU，包括英伟达™（NVIDIA®）Hopper 架构的 FP8 加速支持。

Megatron-Core 提供核心构建模块，如关注机制、转换器模块和层、规范化层和嵌入技术。该库还原生内置了激活重新计算、分布式检查点等附加功能。这些构建模块和功能都经过 GPU 优化，可以采用先进的并行化策略，在英伟达™（NVIDIA®）加速计算基础架构上实现最佳的训练速度和稳定性。Megatron-Core 库的另一个关键组成部分包括先进的模型并行技术（张量、序列、管道、上下文和 MoE 专家并行）。

# 安装设置
推荐使用 NGC's PyTorch container 的最新发布版本配合 DGX 节点使用。如果由于某些原因不能使用这个版本，请使用最新的 PyTorch、CUDA、NCCL 以及 NVIDIA 的 APEX 版本。数据预处理需要 NLTK，虽然这对于训练、评估或下游任务不是必需的。

你可以使用以下 Docker 命令启动一个 PyTorch 容器实例，并挂载 Megatron、你的数据集和检查点：
```
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/megatron:/workspace/megatron -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints nvcr.io/nvidia/pytorch:xx.xx-py3
```

# 使用方法

安装完成后，有几种可能的工作流程，包括：
1. 数据预处理
2. 预训练
3. 微调（对于零样本任务可选）
4. 下游任务评估或文本生成

然而，步骤1和2可以通过使用上述的一个预训练模型来替代。

在 examples 目录中提供了几个用于预训练 BERT 和 GPT 的脚本，以及用于零样本和微调下游任务的脚本，包括 MNLI、RACE、WikiText103 和 LAMBADA 的评估。还有一个用于 GPT 交互式文本生成的脚本。

# 训练
## 数据预处理
训练数据需要预处理。首先，将您的训练数据放在宽松的 json 格式中，每行包含一个文本样本的 json。例如：
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

可以通过使用 [`preprocess_data.py`](./tools/preprocess_data.py) 中的 `--json-key` 标志更改 json 的 `text` 字段名称。其他元数据是可选的，训练中不使用。

将宽松的 json 处理成二进制格式以供训练。使用 `preprocess_data.py` 将 json 转换成 mmap 格式。准备 BERT 训练数据的示例脚本为：
<pre>
python tools/preprocess_data.py \\
       --input my-corpus.json \\
       --output-prefix my-bert \\
       --vocab-file bert-vocab.txt \\
       --tokenizer-type BertWordPieceLowerCase \\
       --split-sentences
</pre>

输出将是两个文件，本例中为 `my-bert_text_sentence.bin` 和 `my-bert_text_sentence.idx`。后续 BERT 训练中指定的 `--data-path` 是完整路径和新文件名，但不包括文件扩展名。

对于 T5 使用与 BERT 相同的预处理，也许将其重命名为：
<pre>
       --output-prefix my-t5 \\
</pre>

GPT 数据预处理需要一些小的修改，即添加合并表，文档结束标记，移除句子拆分，并更改分词器类型：
<pre>
python tools/preprocess_data.py \\
       --input my-corpus.json \\
       --output-prefix my-gpt2 \\
       --vocab-file gpt2-vocab.json \\
       --tokenizer-type GPT2BPETokenizer \\
       --merge-file gpt2-merges.txt \\
       --append-eod
</pre>

这里的输出文件命名为 `my-gpt2_text_document.bin` 和 `my-gpt2_text_document.idx`。如前所述，在 GPT 训练中，使用不带扩展名的更长名称作为 `--data-path`。

更多命令行参数详见源文件 [`preprocess_data.py`](./tools/preprocess_data.py)。

**数据预处理脚本已优化适配，建议使用 `/mnt/data/Codes/json2binidx_tool/megatron/preprocess_data.py` 的脚本。**

## 使用命令行参数执行模型训练脚本

若采用命令行参数传递方式来运行训练脚本，可以参考以下示例脚本编写适配模型的训练脚本：`train_gpt3_175b_distributed.sh`。

### 脚本内容

```bash
#!/bin/bash

# 设定最大CUDA设备连接数
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 定义训练节点和GPU配置
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))

# 指定路径变量
CHECKPOINT_PATH=$1 # <指定检查点存储路径>
TENSORBOARD_LOGS_PATH=$2 # <指定TensorBoard日志路径>
VOCAB_FILE=$3 # <指定词汇文件路径>/gpt2-vocab.json
MERGE_FILE=$4 # <指定合并文件路径>/gpt2-merges.txt
DATA_PATH=$5 # <指定数据路径和文件前缀>_text_document

# 分布式训练参数
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

# GPT模型参数
GPT_MODEL_ARGS=(
    --num-layers 96 
    --hidden-size 12288 
    --num-attention-heads 96 
    --seq-length 2048 
    --max-position-embeddings 2048 
)

# 训练参数
TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1536 
    --rampup-batch-size 16 16 5859375 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

# 模型并行参数
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8 
    --pipeline-model-parallel-size 16 
)

# 数据参数
DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

# 评估和日志参数
EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
```

### 运行脚本

执行以下命令以启动训练：
```bash
bash train_gpt3_175b_distributed.sh
```

## 使用config.yaml运行模型训练脚本

若采用config.yaml形式运行训练脚本，则`config.yaml`可仿照以下示例配置编写：

### 配置内容

```bash
language_model:
  # model architecture
  num_layers: 32
  hidden_size: 8192
  num_attention_heads: 32
  num_query_groups: null

  token_shift_bottleneck_size: 32
  decay_bottleneck_size: 64

  ffn_hidden_size: 28672
  kv_channels: null
  hidden_dropout: 0.0
  attention_dropout: 0.0
  fp32_residual_connection: False

  apply_residual_connection_post_layernorm: False
  layernorm_epsilon: 1.e-5
  layernorm_zero_centered_gamma: False
  add_bias_linear: False
  bias_activation_fusion: False
  add_qkv_bias: False
  gated_linear_unit: True
  activation_func: swiglu
  num_moe_experts: null
  rotary_interleaved: False
  window_size: null

  # initialization
  init_method: null
  init_method_std: 0.02
  output_layer_init_method: null

  # mixed-precision
  apply_query_key_layer_scaling: False
  attention_softmax_in_fp32: False

  # fusion
  bias_swiglu_fusion: True
  masked_softmax_fusion: True
  persist_layer_norm: False
  memory_efficient_layer_norm: False
  bias_dropout_fusion: True
  apply_rope_fusion: True

  # activation recomputation
  recompute_granularity: full
  recompute_method: uniform
  recompute_num_layers: 4
  # recompute_granularity: null
  # recompute_method: null
  # recompute_num_layers: null
  distribute_saved_activations: null

  # fp8 related
  fp8: null
  fp8_margin: 0
  fp8_interval: 1
  fp8_amax_history_len: 1
  fp8_amax_compute_algo: "most_recent"
  fp8_wgrad: True
  activation_func_fp8_input_store: False
  fp8_dot_product_attention: False
  fp8_multi_head_attention: False

  # miscellaneous
  clone_scatter_output_in_embedding: True

  normalization: "LayerNorm"  # alt value supported by TE: "RMSNorm"
  qk_layernorm: False

  # MoE related
  moe_router_load_balancing_type: "aux_loss"
  moe_router_topk: 2
  moe_grouped_gemm: False
  moe_aux_loss_coeff: 0  # 1e-2 would be a good start value for load balance loss.
  moe_z_loss_coeff: null  # 1e-3 would be a good start value for z-loss
  moe_input_jitter_eps: null
  moe_token_dropping: False
  moe_token_dispatcher_type: null
  moe_per_layer_logging: null
  moe_expert_capacity_factor: null
  moe_pad_expert_input_to_capacity: null
  moe_token_drop_policy: null
  moe_layer_recompute: null
  
  test_mode: False
  calculate_per_token_loss: False
  disable_parameter_transpose_cache: null

model_parallel:
  # Model parallelism
  tensor_model_parallel_size: 8
  context_parallel_size: 1
  pipeline_model_parallel_size: 1
  virtual_pipeline_model_parallel_size: null
  sequence_parallel: True
  expert_model_parallel_size: 1
  moe_extended_tp: False
  encoder_pipeline_model_parallel_size: null
  # Initialization
  perform_initialization: True
  use_cpu_initialization: False

  # Training
  fp16: False
  bf16: True
  params_dtype: null # Set from above arguments for core
  timers: null

  # Optimizations
  gradient_accumulation_fusion: True
  async_tensor_model_parallel_allreduce: True
  tp_comm_overlap: False
  tp_comm_overlap_ag: False
  tp_comm_overlap_rs: False
  tp_comm_overlap_rs_dgrad: False
  deterministic_mode: False
  cross_entropy_loss_fusion: False
  enable_cuda_graph: False
  defer_embedding_wgrad_compute: False
  wgrad_deferral_limit: 0

  # Debug Options
  tp_comm_split_ag: True
  tp_comm_atomic_ag: True
  tp_comm_split_rs: True
  tp_comm_atomic_rs: True
  tp_comm_bulk_wgrad: True
  tp_comm_bulk_dgrad: True

  # Parallelism
  finalize_model_grads_func: null

  # Pipeline Parallel
  pipeline_dtype: null
  grad_scale_func: null
  enable_autocast: False
  autocast_dtype: null
  variable_seq_lengths: False
  num_microbatches_with_partial_activation_checkpoints: null
  overlap_p2p_comm: False
  batch_p2p_comm: True
  batch_p2p_sync: True
  use_ring_exchange_p2p: False
  deallocate_pipeline_outputs: False
  no_sync_func: null
  grad_sync_func: null
  param_sync_func: null
  pipeline_model_parallel_split_rank: null

  # CPU Offloading
  cpu_offloading: False
  cpu_offloading_num_layers: 0
  _cpu_offloading_context: null
  cpu_offloading_weights: False
  cpu_offloading_activations: True

  # Timing
  barrier_with_L1_time: True

  use_te_rng_tracker: False

# training:
use_legacy_models: False
spec: null
micro_batch_size: 1
global_batch_size: 64
rampup_batch_size: null
check_for_nan_in_loss_and_grad: True
num_layers_per_virtual_pipeline_stage: null

encoder_num_layers: null
decoder_num_layers: null
rotary_seq_len_interpolation_factor: null
add_position_embedding: False
make_vocab_size_divisible_by: 128
group_query_attention: False

vision_pretraining: False
vision_pretraining_type: classify

exit_signal_handler: False
exit_duration_in_mins: null
exit_interval: null

untie_embeddings_and_output_weights: True
position_embedding_type: none
rotary_percent: 0.5
openai_gelu: False
squared_relu: False
swiglu: True
onnx_safe: null
bert_binary_head: True
max_position_embeddings: 4096

transformer_impl: local
use_flash_attn: False
seed: 1234
data_parallel_random_init: False

# Optimizer
optimizer: adam
lr: 2.5e-4
lr_decay_style: WSD
lr_decay_iters: null
lr_decay_samples: null
lr_warmup_fraction: 0.005
lr_warmup_iters: 0
lr_warmup_samples: 0
lr_warmup_init: 0.0
lr_wsd_decay_iters: 900
lr_wsd_decay_style: cosine
min_lr: 2.5e-5
decoupled_lr: null
weight_decay: 0.1
start_weight_decay: null
end_weight_decay: null
weight_decay_incr_style: constant
clip_grad: 1.0
adam_beta1: 0.9
adam_beta2: 0.995
adam_eps: 1.e-08
sgd_momentum: 0.9
override_opt_param_scheduler: False
use_checkpoint_opt_param_scheduler: False

# checkpointing arguments
save: /mnt/data/Outs/rwkv6-megatron-wandb/ckpt
save_interval: 2000
no_save_optim: null
no_save_rng: null
load: null
no_load_optim: null
no_load_rng: null
pretrained_checkpoint: null
finetune: False
use_checkpoint_args: False
exit_on_missing_checkpoint: False

# loss arguments
loss_scale: null
initial_loss_scale: 4294967296
min_loss_scale: 1.0
loss_scale_window: 1000 
hysteresis: 2
accumulate_allreduce_grads_in_fp32: False
fp16_lm_cross_entropy: False

# distributed arguments
distributed_backend: nccl
distributed_timeout_minutes: 100
overlap_grad_reduce: False
delay_grad_reduce: True
overlap_param_gather: False
delay_param_gather: False
scatter_gather_tensors_in_pipeline: True
local_rank: null
lazy_mpu_init: null
empty_unused_memory_level: 0
standalone_embedding_stage: False
use_distributed_optimizer: True
nccl_communicator_config_path: null
use_tp_pp_dp_mapping: False
ddp_bucket_size: null
ddp_average_in_collective: True
check_weight_hash_across_dp_replicas_interval: 128

train_iters: 99000
eval_iters: 32
eval_interval: 128
skip_train: False
auto_train_iters: True

adlr_autoresume: False
adlr_autoresume_interval: 1000

# garbage collection
manual_gc: False
manual_gc_interval: 0
manual_gc_eval: True

tp_comm_overlap_cfg: null

#data
data_path:
  - /mnt/data/Datasets/pretrain_data/pretrain_text_document
split: '99,1,0'
train_data_path: null
valid_data_path: null
test_data_path: null
data_cache_path: /mnt/data/Logs/rwkv6-megatron-wandb/test1-data-cache
num_dataset_builder_threads: 2
mock_data: False
vocab_file: null
merge_file: null
vocab_extra_ids: 0
seq_length: 4096
encoder_seq_length: null
decoder_seq_length: null
retriever_seq_length: 256
sample_rate: 1.0
mask_prob: 0.15
short_seq_prob: 0.1
num_workers: 2
mmap_bin_files: True
reset_position_ids: False
reset_attention_mask: False
eod_mask_loss: False
create_attention_mask_in_dataloader: False
train_samples: null
dataloader_type: null
vocab_size: null
tokenizer_type: HuggingFaceTokenizer
tokenizer_model: /mnt/data/Models/Huggingface/Mobius-r6-chat-CHNtuned-12b-16k-v5-5
retro_project_dir: null
retro_cyclic_train_iters: null
#profile:
profile: False
profile_ranks: [0]
profile_step_end: 2
profile_step_start: 1

# logging:
logging_level: null
log_params_norm: False
log_num_zeros_in_grad: False
log_throughput: True
log_progress: True
timing_log_level: 0
timing_log_option: minmax
tensorboard_dir: /mnt/data/Logs/tensorboard
tensorboard_log_interval: 1
tensorboard_queue_size: 1000
log_timers_to_tensorboard: True
log_batch_size_to_tensorboard: True
log_learning_rate_to_tensorboard: True
log_validation_ppl_to_tensorboard: True
log_memory_to_tensorboard: False
log_world_size_to_tensorboard: True
log_loss_scale_to_tensorboard: True
wandb_project: ''
wandb_exp_name: ''
wandb_save_dir: ''
enable_one_logger: False
one_logger_project: e2e-tracking
one_logger_entity: hwinf_dcm
one_logger_run_name: null
log_interval: 1

log_straggler: False
straggler_minmax_count: 1
# TODO: some more for straggler detection

async_save: False
use_dist_ckpt: True
dist_ckpt_format: torch_dist
ckpt_fully_parallel_save: True
ckpt_assume_constant_structure: null
non_persistent_save_interval: null

debug_dump: False

load_pretrained_rwkv: /mnt/data/Models/rwkv-r6-30b-init.pth
load_pretrained_rwkv_source_type: rwkv-lm

check_against_pretrained: False
```
### 编写脚本
编写以下脚本配置环境变量与配置文件：

```bash
#!/bin/bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=24
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc-per-node 8 --nnodes 1 pretrain.py --yaml-cfg config.yaml | tee output.log
```

### 运行脚本
执行以下脚本以启动训练：
```bash
bash train.sh
```

## 命令行参数传参说明：
如果你使用[命令行参数运行训练脚本](#使用命令行参数执行模型训练脚本)，以下是配置参数的详细说明：

### 参数组：Transformer Engine

#### `--fp8-format`
- **描述**: 选择在前向和后向传递中使用的 FP8 格式。
- **可选值**: 
  - `e4m3`: 使用 e4m3 方案。
  - `hybrid`: 使用混合方案。
- **默认值**: `None`

#### `--fp8-margin`
- **描述**: FP8 计算中的缩放余量。
- **类型**: 整数。
- **默认值**: `0`

#### `--fp8-interval`
- **描述**: FP8 缩放更新间隔。
- **类型**: 整数。
- **默认值**: `1`

#### `--fp8-amax-history-len`
- **描述**: 记录每个张量的最大绝对值（amax）历史的步数。
- **类型**: 整数。
- **默认值**: `1`

#### `--fp8-amax-compute-algo`
- **描述**: 从历史数据中计算 amax 的算法。
- **可选值**:
  - `most_recent`: 使用最近的值。
  - `max`: 使用历史中的最大值。
- **默认值**: `most_recent`

#### `--no-fp8-wgrad`
- **描述**: 即使在 FP8 运行中，也以更高的精度执行权重梯度计算。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`（因为默认行为是不添加此参数，故为 `True`）

#### `--transformer-impl`
- **描述**: 选择使用哪种 Transformer 实现。
- **可选值**:
  - `local`: 使用本地实现。
  - `transformer_engine`: 使用 transformer engine。
- **默认值**: `transformer_engine`

### 参数组：inference

#### `--inference-batch-times-seqlen-threshold`
- **描述**: 在推理时，如果批处理大小与序列长度的乘积小于此阈值，则不使用流水线处理，否则将使用。
- **类型**: 整数。
- **默认值**: `512`

#### `--max-tokens-to-oom`
- **描述**: 推理过程中的最大令牌数量。这里的令牌包括提示中的令牌数量加上要生成的令牌数量。该设置可以在服务器出现内存不足之前抛出错误。
- **类型**: 整数。
- **默认值**: `12000`

#### `--output-bert-embeddings`
- **描述**: 从模型中输出 Bert 嵌入（通过均值池化），而不是其二进制头输出或整个隐藏批量。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--bert-embedder-type`
- **描述**: 选择 Megatron 或 Huggingface 作为 Bert 嵌入器。
- **可选值**:
  - `megatron`: 使用 Megatron。
  - `huggingface`: 使用 Huggingface。
- **默认值**: `megatron`

### 参数组：retro

#### `--retro-project-dir`
- **描述**: Retro 项目目录，包含预训练的预处理数据。此目录在预处理期间建立（参见 `tools/retro/README.md`），包含用于块数据库和预训练邻居的子目录。
- **默认值**: `None`

#### `--retro-add-retriever`
- **描述**: 在 Transformer 中添加检索器，用于预训练 Retro 模型。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--retro-cyclic-train-iters`
- **描述**: 设置循环 Retro 训练的迭代次数。
- **类型**: 整数。
- **默认值**: `None`

#### `--retro-encoder-layers`
- **描述**: 用于检索编码器的层数。
- **类型**: 整数。
- **默认值**: `2`

#### `--retro-encoder-hidden-dropout`
- **描述**: 检索编码器的隐藏层丢弃率。
- **类型**: 浮点数。
- **默认值**: `0.1`

#### `--retro-encoder-attention-dropout`
- **描述**: 检索编码器的注意力层丢弃率。
- **类型**: 浮点数。
- **默认值**: `0.1`

#### `--retro-num-neighbors`
- **描述**: 预训练期间检索的邻居数量。
- **类型**: 整数。
- **默认值**: `2`

#### `--retro-num-retrieved-chunks`
- **描述**: 从检索数据库中检索的块数量。
- **类型**: 整数。
- **默认值**: `2`

#### `--retro-attention-gate`
- **描述**: 门控交叉注意力。
- **类型**: 浮点数。
- **默认值**: `1`

#### `--retro-no-verify-neighbor-count`
- **描述**: 跳过验证 GPT 数据集的长度是否等于保存的邻居数量。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`（因为默认行为是不添加此参数，故为 `True`）
  
### 参数组：network size

#### `--num-layers`
- **描述**: Transformer的层数。
- **类型**: 整数。
- **默认值**: `None`

#### `--encoder-num-layers`
- **描述**: 编码器的Transformer层数。
- **类型**: 整数。
- **默认值**: `None`

#### `--decoder-num-layers`
- **描述**: 解码器的Transformer层数。
- **类型**: 整数。
- **默认值**: `None`

#### `--hidden-size`
- **描述**: Transformer隐藏层的大小。
- **类型**: 整数。
- **默认值**: `None`

#### `--ffn-hidden-size`
- **描述**: Transformer前馈网络的隐藏层大小。如果未提供，则默认设置为隐藏大小的4倍。
- **类型**: 整数。
- **默认值**: `None`

#### `--num-attention-heads`
- **描述**: Transformer的注意力头数。
- **类型**: 整数。
- **默认值**: `None`

#### `--kv-channels`
- **描述**: 多头注意力中投影权重的维度。如果未提供，则设置为 `args.hidden_size // args.num_attention_heads`。
- **类型**: 整数。
- **默认值**: `None`

#### `--group-query-attention`
- **描述**: 使用分组查询注意力。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--num-query-groups`
- **描述**: 查询分组的数量。
- **类型**: 整数。
- **默认值**: `1`

#### `--max-position-embeddings`
- **描述**: 使用的最大位置嵌入数量。这是位置嵌入的大小。
- **类型**: 整数。
- **默认值**: `None`

#### `--position-embedding-type`
- **描述**: 位置嵌入类型。
- **可选值**:
  - `learned_absolute`: 学习的绝对位置嵌入。
  - `rope`: 循环位置嵌入。
  - `none`: 不使用位置嵌入。
- **默认值**: `learned_absolute`

#### `--use-rotary-position-embeddings`
- **描述**: 是否使用旋转位置嵌入。已弃用：使用 `--position-embedding-type`。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--rotary-base`
- **描述**: 用于旋转位置嵌入的基数，默认为10000。
- **类型**: 整数。
- **默认值**: `10000`

#### `--rotary-percent`
- **描述**: 使用的旋转维度的百分比，默认为100%。
- **类型**: 浮点数。
- **默认值**: `1.0`

#### `--rotary-interleaved`
- **描述**: 使用交错的旋转嵌入。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--rotary-seq-len-interpolation-factor`
- **描述**: 旋转嵌入的序列长度插值因子。
- **类型**: 整数。
- **默认值**: `None`

#### `--no-position-embedding`
- **描述**: 禁用位置嵌入。已弃用：使用 `--position-embedding-type`。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`（因为默认行为是不添加此参数，故为 `True`）

#### `--make-vocab-size-divisible-by`
- **描述**: 使词汇大小可被此值整除。添加此设置是出于计算效率的考虑。
- **类型**: 整数。
- **默认值**: `128`

#### `--normalization`
- **描述**: 使用的标准化技术。
- **可选值**:
  - `LayerNorm`: 层归一化。
  - `RMSNorm`: 均方根归一化。
- **默认值**: `LayerNorm`

#### `--norm-epsilon`
- **描述**: 层归一化和均方根归一化的 epsilon 值。
- **类型**: 浮点数。
- **默认值**: `1e-5`

#### `--apply-layernorm-1p`
- **描述**: 调整层归一化权重，使其围绕零中心。这可以提高数值稳定性。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--apply-residual-connection-post-layernorm`
- **描述**: 如果设置，则使用原始 BERT 残差连接排序。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--openai-gelu`
- **描述**: 使用 OpenAI 的 GeLU 实现。除非出于向后兼容的原因，否则不应使用此选项。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--squared-relu`
- **描述**: 使用平方 relu 激活函数代替默认的 gelu。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--swiglu`
- **描述**: 使用门控线性单元和 SiLU 激活函数代替默认的 gelu。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--onnx-safe`
- **描述**: 使用针对 Torch ONNX 导出器已知问题的解决方案。
- **类型**: 布尔值。
- **默认值**: `False`

#### `--bert-no-binary-head`
- **描述**: 禁用 BERT 二进制头。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`（因为默认行为是不添加此参数，故为 `True`）

#### `--untie-embeddings-and-output-weights`
- **描述**: 解绑嵌入和输出权重。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`
  
### 参数组：straggler

#### `--log-straggler`
- **描述**: 如果设置，将跟踪并记录每个 GPU 的滞后者。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--disable-straggler-on-startup`
- **描述**: 如果设置，启动时禁用 StragglerDetector。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--straggler-ctrlr-port`
- **描述**: 在运行时开关 StragglerDetector 的端口号。
- **类型**: 整数。
- **默认值**: `65535`

#### `--straggler-minmax-count`
- **描述**: 报告高/低估计吞吐量的等级数量。
- **类型**: 整数。
- **默认值**: `1`

### 参数组：one logger

#### `--no-one-logger`
- **描述**: 如果设置，禁用 one_logger 来跟踪 E2E 指标。请注意，one_logger 是一个内部工具，不对外提供。安装详细信息请访问 [NVIDIA Confluence](https://confluence.nvidia.com/display/MLWFO/Package+Repositories)。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`（因为默认行为是不添加此参数，故为 `True`）

#### `--one-logger-project`
- **描述**: one_logger 的项目名称。如果设置了 `--no-one-logger`，将忽略此设置。
- **类型**: 字符串。
- **默认值**: `megatron-lm`

#### `--one-logger-run-name`
- **描述**: 显示的 one_logger 运行名称。如果设置了 `--no-one-logger`，将忽略此设置。
- **类型**: 字符串。
- **默认值**: `None`

#### `--one-logger-async`
- **描述**: 如果设置，强制 one_logger 使用异步模式。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--app-tag-run-name`
- **描述**: 属于同一训练运行的作业应该具有相同的名称。它将用于跟踪通过多个不同作业完成的训练的进度。
- **类型**: 字符串。
- **默认值**: `None`

#### `--app-tag-run-version`
- **描述**: 当前作业所属的训练版本。它将用于跟踪可能改变性能基线的应用程序端的更改。
- **类型**: 字符串。
- **默认值**: `0.0.0`
  
### 参数组：logging

#### `--log-params-norm`
- **描述**: 如果设置，计算并记录参数的范数。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--log-num-zeros-in-grad`
- **描述**: 如果设置，计算并记录梯度中零的数量。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--log-throughput`
- **描述**: 如果设置，计算并记录每个 GPU 的吞吐量。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--log-progress`
- **描述**: 如果设置，记录进度（以处理的令牌数和浮点运算数为单位）到检查点目录中的 progress.txt 文件。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--timing-log-level`
- **描述**: 测量和报告时间的粒度级别。
- **类型**: 整数。
- **默认值**: `0`
- **可选值**:
  - `0`: 只报告迭代时间，确保计时不引入额外开销。
  - `1`: 报告每次迭代中执行次数非常有限的操作（如梯度全归约）的计时。
  - `2`: 报告可能在每次迭代中执行多次的操作的计时。

#### `--no-barrier-with-level-1-timing`
- **描述**: 如果未设置，使用 level 1 时间测量的屏障。请注意，由用户确保调用屏障与他们的计时器不会导致挂起。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--timing-log-option`
- **描述**: 日志计时选项。
- **类型**: 字符串。
- **默认值**: `minmax`
- **可选值**:
  - `max`: 报告所有等级中的最大时间。
  - `minmax`: 报告所有等级中的最小和最大时间。
  - `all`: 报告所有等级的时间。

#### `--tensorboard-log-interval`
- **描述**: 向 tensorboard 报告的间隔。
- **类型**: 整数。
- **默认值**: `1`

#### `--tensorboard-queue-size`
- **描述**: tensorboard 的待处理事件和摘要队列大小，之后的一个 ‘add’ 调用会强制刷新到磁盘。
- **类型**: 整数。
- **默认值**: `1000`

#### `--log-timers-to-tensorboard`
- **描述**: 如果设置，将计时器写入 tensorboard。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--log-batch-size-to-tensorboard`
- **描述**: 如果设置，将批量大小写入 tensorboard。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--no-log-learnig-rate-to-tensorboard`
- **描述**: 禁用将学习率记录到 tensorboard。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--no-log-loss-scale-to-tensorboard`
- **描述**: 禁用将损失比例记录到 tensorboard。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--log-validation-ppl-to-tensorboard`
- **描述**: 如果设置，将验证困惑度写入 tensorboard。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--log-memory-to-tensorboard`
- **描述**: 启用将内存日志记录到 tensorboard。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--log-world-size-to-tensorboard`
- **描述**: 启用将世界大小记录到 tensorboard。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--wandb-project`
- **描述**: wandb 项目名称。默认情况下忽略 wandb。
- **类型**: 字符串。
- **默认值**: `''`

#### `--wandb-exp-name`
- **描述**: wandb 实验名称。
- **类型**: 字符串。
- **默认值**: `''`

#### `--wandb-save-dir`
- **描述**: 保存 wandb 结果的本地路径。
- **类型**: 字符串。
- **默认值**: `''`

#### `--logging-level`
- **描述**: 设置默认日志级别。
- **类型**: 整数。
- **默认值**: `None`

### 参数组：regularization

#### `--attention-dropout`
- **描述**: 注意力层之后的dropout概率。
- **类型**: 浮点数。
- **默认值**: `0.1`

#### `--hidden-dropout`
- **描述**: 隐藏状态Transformer的dropout概率。
- **类型**: 浮点数。
- **默认值**: `0.1`

#### `--weight-decay`
- **描述**: L2正则化的权重衰减系数。
- **类型**: 浮点数。
- **默认值**: `0.01`

#### `--start-weight-decay`
- **描述**: L2正则化的初始权重衰减系数。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--end-weight-decay`
- **描述**: 运行结束时L2正则化的权重衰减系数。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--weight-decay-incr-style`
- **描述**: 权重衰减增量函数。
- **类型**: 字符串。
- **默认值**: `constant`
- **可选值**:
  - `constant`: 常数增量。
  - `linear`: 线性增量。
  - `cosine`: 余弦增量。

#### `--clip-grad`
- **描述**: 基于全局L2范数的梯度裁剪。
- **类型**: 浮点数。
- **默认值**: `1.0`

#### `--adam-beta1`
- **描述**: 计算梯度及其平方运行平均的第一系数。
- **类型**: 浮点数。
- **默认值**: `0.9`

#### `--adam-beta2`
- **描述**: 计算梯度及其平方运行平均的第二系数。
- **类型**: 浮点数。
- **默认值**: `0.999`

#### `--adam-eps`
- **描述**: 为了提高数值稳定性，添加到分母的项。
- **类型**: 浮点数。
- **默认值**: `1e-08`

#### `--sgd-momentum`
- **描述**: SGD的动量因子。
- **类型**: 浮点数。
- **默认值**: `0.9`

### 参数组：training

#### `--micro-batch-size`
- **描述**: 每个模型实例的批次大小（局部批次大小）。全局批次大小是局部批次大小乘以数据并行大小乘以微批次数量。
- **类型**: 整数。
- **默认值**: `None`

#### `--batch-size`
- **描述**: 旧批次大小参数，不推荐使用。使用 `--micro-batch-size` 替代。
- **类型**: 整数。
- **默认值**: `None`

#### `--global-batch-size`
- **描述**: 训练批次大小。如果设置，它应该是微批次大小乘以数据并行大小的倍数。如果此值为 `None`，则使用微批次大小乘以数据并行大小作为全局批次大小。
- **类型**: 整数。
- **默认值**: `None`

#### `--rampup-batch-size`
- **描述**: 批次大小增加，使用以下值：`<start batch size> <batch size increment> <ramp-up samples>`。例如，使用 `--rampup-batch-size 16 8 300000 --global-batch-size 1024` 将从全局批次大小16开始，在126个间隔内线性增加批次大小至1024。在每个间隔中将使用约2380个样本。
- **类型**: 列表。
- **默认值**: `None`

#### `--recompute-activations`
- **描述**: 重新计算激活以允许使用更大的模型、序列和批次大小进行训练。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--recompute-granularity`
- **描述**: 为了允许使用更大的模型、序列和批次大小进行训练，保存激活。支持两种粒度：1) 全部：整个变压器层被重新计算，2) 选择性：变压器层的核心注意部分被重新计算。
- **类型**: 字符串。
- **默认值**: `None`
- **可选值**:
  - `full`: 整个变压器层被重新计算。
  - `selective`: 变压器层的核心注意部分被重新计算。

#### `--no-check-for-nan-in-loss-and-grad`
- **描述**: 不检查损失和梯度中的 NaN 值。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--distribute-saved-activations`
- **描述**: 如果设置，将重新计算的激活分布到模型并行组。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--recompute-method`
- **描述**: 1) 均匀：均匀划分总变压器层数并在指定的粒度下重新计算每个划分块的输入激活，2) 块：每个流水线阶段仅重新计算一组变压器层的输入激活，并在指定的粒度下进行其余部分的计算，不重新计算。
- **类型**: 字符串。
- **默认值**: `None`
- **可选值**:
  - `uniform`: 均匀划分。
  - `block`: 块划分。

#### `--recompute-num-layers`
- **描述**: 1) 均匀：每个均匀划分的重新计算单元中的变压器层数，2) 块：每个流水线阶段中重新计算的个别变压器层的数量。
- **类型**: 整数。
- **默认值**: `None`

#### `--no-clone-scatter-output-in-embedding`
- **描述**: 如果未设置，克隆嵌入层中散列输出的输出以原始张量。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--profile`
- **描述**: 启用 nsys 分析。使用此选项时，nsys 选项应在命令行中指定。一个例子 `nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop`。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--profile-step-start`
- **描述**: 开始分析的全局步骤。
- **类型**: 整数。
- **默认值**: `10`

#### `--profile-step-end`
- **描述**: 停止分析的全局步骤。
- **类型**: 整数。
- **默认值**: `12`

#### `--profile-ranks`
- **描述**: 进行分析的全局排名。
- **类型**: 列表。
- **默认值**: `[0]`

#### `--tp-comm-overlap`
- **描述**: 启用张量并行通信和 GEMM 核心的重叠。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--tp-comm-overlap-cfg`
- **描述**: 启用 tp_comm_overlap 时的配置文件。
- **类型**: 字符串。
- **默认值**: `None`

#### `--disable-tp-comm-overlap-ag`
- **描述**: 禁用与 GEMM 重叠的 All-Gather。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--disable-tp-comm-overlap-rs`
- **描述**: 禁用与 GEMM 重叠的 Reduce-Scatter。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--tp-comm-overlap-rs-dgrad`
- **描述**: 启用与 dgrad GEMM 重叠的 Reduce-Scatter。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--disable-tp-comm-bulk-dgrad`
- **描述**: 禁用与 bprop 激活梯度 GEMM 重叠的 All-Gather。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--disable-tp-comm-bulk-wgrad`
- **描述**: 禁用与 bprop 权重梯度 GEMM 重叠的 Reduce-Scatter。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--use-cpu-initialization`
- **描述**: 如果设置，初始化权重在 CPU 上。这消除了基于张量并行性的初始化差异。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `None`

#### `--empty-unused-memory-level`
- **描述**: 每次迭代（训练和评估）调用 `torch.cuda.empty_cache()` 来减少碎片。
- **类型**: 整数。
- **默认值**: `0`
- **可选值**:
  - `0`: 关闭。
  - `1`: 中等。
  - `2`: 激进。

#### `--deterministic-mode`
- **描述**: 选择具有确定性执行的代码。这通常意味着执行速度较慢，但适合调试和测试。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--check-weight-hash-across-dp-replicas-interval`
- **描述**: 检查跨数据并行副本的权重哈希是否相同的间隔。如果未指定，则不检查权重哈希。
- **类型**: 整数。
- **默认值**: `None`

#### `--calculate-per-token-loss`
- **描述**: 根据全局批次中的非填充令牌数量对交叉熵损失进行缩放，与假设所有令牌都是非填充的默认行为相反。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

# Deprecated
#### `--checkpoint-activations` 
- **描述**: 保存激活状态以允许使用更大的模型、序列和批次大小进行训练。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--train-iters`
- **描述**: 在所有训练运行中训练的总迭代次数。注意，应提供 `train-iters` 或 `train-samples` 其中之一。
- **类型**: 整数。
- **默认值**: `None`

#### `--train-samples`
- **描述**: 在所有训练运行中训练的总样本数。注意，应提供 `train-iters` 或 `train-samples` 其中之一。
- **类型**: 整数。
- **默认值**: `None`

#### `--log-interval`
- **描述**: 报告损失和时间间隔。
- **类型**: 整数。
- **默认值**: `100`

#### `--exit-interval`
- **描述**: 程序在迭代次数可被此值整除时退出。
- **类型**: 整数。
- **默认值**: `None`

#### `--exit-duration-in-mins`
- **描述**: 程序在此分钟数后退出。
- **类型**: 整数。
- **默认值**: `None`

#### `--exit-signal-handler`
- **描述**: 如果接收到 SIGTERM，动态保存检查点并关闭训练。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--tensorboard-dir`
- **描述**: 将 TensorBoard 日志写入此目录。
- **类型**: 字符串。
- **默认值**: `None`

#### `--no-masked-softmax-fusion`
- **描述**: 禁用查询键值缩放、掩蔽和 softmax 的融合。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--no-bias-gelu-fusion`
- **描述**: 禁用偏置和 gelu 融合。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--no-bias-swiglu-fusion`
- **描述**: 禁用偏置和 swiglu 融合，仅在使用 megatron-core 时可用。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--no-bias-dropout-fusion`
- **描述**: 禁用偏置和 dropout 融合。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--no-rope-fusion`
- **描述**: 禁用 rope 融合，仅在使用 megatron-core 时可用。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--cross-entropy-loss-fusion`
- **描述**: 启用交叉熵损失计算的融合。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--use-flash-attn`
- **描述**: 使用 FlashAttention 实现注意力。参考文献 [FlashAttention Paper](https://arxiv.org/abs/2205.14135)。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--disable-bias-linear`
- **描述**: 禁用线性层中的偏置。
- **类型**: 布尔值（使用 `store_false` 表表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--add-qkv-bias`
- **描述**: 仅在 QKV 线性层中启用偏置。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--optimizer`
- **描述**: 选择优化器函数。
- **类型**: 字符串。
- **默认值**: `adam`
- **可选值**:
  - `adam`
  - `sgd`

#### `--dataloader-type`
- **描述**: 选择数据加载器类型。
- **类型**: 字符串。
- **默认值**: `None`
- **可选值**:
  - `single`
  - `cyclic`
  - `external`

#### `--no-async-tensor-model-parallel-allreduce` # Deprecated
- **描述**: 此标志已弃用且被忽略。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--no-persist-layer-norm`
- **描述**: 禁用使用持久化融合层归一化内核。此内核仅支持一组隐藏大小。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--sequence-parallel`
- **描述**: 启用序列并行优化。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--no-gradient-accumulation-fusion`
- **描述**: 禁用将梯度累积与线性层的权重梯度计算融合。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--use-mcore-models` # Deprecated
- **描述**: 此选项已废弃。使用 megatron core 的实现。现在忽略此选项，mcore 模型是默认设置，如果不想使用 core 模型，请使用 `--use-legacy-models`。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--use-legacy-models`
- **描述**: 使用旧版 Megatron 模型，而非 Megatron-Core 模型。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--manual-gc`
- **描述**: 禁用基于阈值的默认垃圾收集器，并手动触发垃圾收集。手动垃圾收集有助于对齐各等级的收集时机，减少 CPU 相关抖动的影响。启用手动 gc 时，默认只在验证程序的开始和结束时执行垃圾收集。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--manual-gc-interval`
- **描述**: 触发手动垃圾收集的训练步间隔。当值设置为 0 时，训练步之间不触发垃圾收集。
- **类型**: 整数。
- **默认值**: `0`

#### `--no-manual-gc-eval`
- **描述**: 使用手动垃圾收集时，在每次评估运行的开始和结束时禁用垃圾收集。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--disable-tp-comm-split-ag`
- **描述**: 禁用与 fprop GEMM 重叠的 All-Gather。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--disable-tp-comm-split-rs`
- **描述**: 禁用与 fprop GEMM 重叠的 Reduce-Scatter。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

### 参数组：initialization

#### `--seed`
- **描述**: 用于 Python、NumPy、PyTorch 和 CUDA 的随机种子。
- **类型**: 整数。
- **默认值**: `1234`

#### `--data-parallel-random-init`
- **描述**: 启用跨数据并行等级的参数的随机初始化。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--init-method-std`
- **描述**: 用于权重初始化的零均值正态分布的标准差。
- **类型**: 浮点数。
- **默认值**: `0.02`

#### `--init-method-xavier-uniform`
- **描述**: 启用 Xavier 均匀参数初始化。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

### 参数组：learning rate

#### `--lr`
- **描述**: 初始学习率。根据衰减样式和初始热身，每次迭代的学习率会有所不同。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--lr-decay-style`
- **描述**: 学习率衰减函数。
- **类型**: 字符串。
- **默认值**: `linear`
- **可选值**:
  - `constant`
  - `linear`
  - `cosine`
  - `inverse-square-root`
  - `WSD`

#### `--lr-wsd-decay-style`
- **描述**: WSD 退火阶段的衰减样式。
- **类型**: 字符串。
- **默认值**: `exponential`
- **可选值**:
  - `exponential`
  - `linear`
  - `cosine`

#### `--lr-decay-iters`
- **描述**: 学习率衰减的迭代次数，如果为 `None` 则默认为 `--train-iters`。
- **类型**: 整数。
- **默认值**: `None`

#### `--lr-decay-samples`
- **描述**: 学习率衰减的样本数，如果为 `None` 则默认为 `--train-samples`。
- **类型**: 整数。
- **默认值**: `None`

#### `--lr-wsd-decay-samples`
- **描述**: WSD 调度中退火阶段的样本数。
- **类型**: 整数。
- **默认值**: `None`

#### `--lr-wsd-decay-iters`
- **描述**: WSD 调度中退火阶段的迭代次数。
- **类型**: 整数。
- **默认值**: `None`

#### `--lr-warmup-fraction`
- **描述**: 用于热身的 `lr-warmup-(iters/samples)` 的分数（浮点数形式）。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--lr-warmup-iters`
- **描述**: 线性热身学习率的迭代次数。
- **类型**: 整数。
- **默认值**: `0`

#### `--lr-warmup-samples`
- **描述**: 线性热身学习率的样本数。
- **类型**: 整数。
- **默认值**: `0`

#### `--lr-warmup-init`
- **描述**: 学习率热身的初始值。调度器从此值开始热身。
- **类型**: 浮点数。
- **默认值**: `0.0`

#### `--warmup`
- **描述**: 旧的学习率热身参数，不推荐使用。请使用上面的 `--lr-warmup-*` 参数之一。
- **类型**: 整数。
- **默认值**: `None`

#### `--min-lr`
- **描述**: 学习率的最小值。调度器将剪切低于此阈值的值。
- **类型**: 浮点数。
- **默认值**: `0.0`

#### `--override-opt_param-scheduler`
- **描述**: 重置调度器的值（学习率、热身迭代次数、最小学习率、最大迭代次数和衰减样式），忽略检查点中的值。注意，以上所有值都将被重置。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--use-checkpoint-opt_param-scheduler`
- **描述**: 使用检查点设置调度器的值（学习率、热身迭代次数、最小学习率、最大迭代次数和衰减样式），忽略输入参数。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--decoupled-lr`
- **描述**: 输入层和输出层的单独学习率。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--decoupled-min-lr`
- **描述**: 输入层和输出层的学习率的最小值。调度器将剪切低于此阈值的值。
- **类型**: 浮点数。
- **默认值**: `None`

### 参数组：checkpointing

#### `--save`
- **描述**: 保存检查点的输出目录。
- **类型**: 字符串。
- **默认值**: `None`

#### `--save-interval`, `--persistent-save-interval`
- **描述**: 持久检查点保存之间的迭代次数。
- **类型**: 整数。
- **默认值**: `None`

#### `--no-save-optim`
- **描述**: 不保存当前优化器。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--no-save-rng`
- **描述**: 不保存当前的随机数生成器状态。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--load`
- **描述**: 包含模型检查点的目录。
- **类型**: 字符串。
- **默认值**: `None`

#### `--no-load-optim`
- **描述**: 加载检查点时不加载优化器。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--no-load-rng`
- **描述**: 加载检查点时不加载随机数生成器状态。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--non-persistent-save-interval`
- **描述**: 非持久保存之间的迭代次数。
- **类型**: 整数。
- **默认值**: `None`

#### `--non-persistent-ckpt-type`
- **描述**: 非持久模型检查点的类型。"global" - 保存为标准检查点，删除旧检查点；"local" - 每个等级本地保存检查点的一部分；"in_memory" - 一种特殊的本地检查点，避免序列化；None - 不进行非持久检查点（默认选项）。
- **类型**: 字符串。
- **可选值**:
  - `global`
  - `local`
  - `in_memory`
  - `None`
- **默认值**: `None`

#### `--non-persistent-global-ckpt-dir`
- **描述**: 包含全局非持久模型检查点的目录。
- **类型**: 字符串。
- **默认值**: `None`

#### `--finetune`
- **描述**: 加载模型进行微调。不加载检查点中的优化器或随机数生成器状态，并将迭代设置为0。假设在加载发布检查点时使用。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--pretrained-checkpoint`
- **描述**: 包含用于微调的预训练模型检查点的目录。
- **类型**: 字符串。
- **默认值**: `None`

#### `--ckpt-step`
- **描述**: 加载模型的检查点步骤。
- **类型**: 整数。
- **默认值**: `None`

#### `--no-initialization`
- **描述**: 构建模型时不进行初始化，可以减少启动时间，当确定从检查点加载时使用。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--use-checkpoint-args`
- **描述**: 用检查点中的参数覆盖任何命令行参数。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--exit-on-missing-checkpoint`
- **描述**: 如果设置了 `--load`，但找不到检查点（例如，路径错误），则退出而不是随机初始化。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--use-dist-ckpt`
- **描述**: 使用分布式检查点格式。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--auto-detect-ckpt-format`
- **描述**: 确定检查点格式是传统格式还是分布式格式。如果为 `False`，则仅在设置了 `args.use_dist_ckpt` 时预期分布式检查点。可能稍微延迟加载。
- **类型**: 布尔值（使用 `store_true` 表表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--dist-ckpt-format`
- **描述**: 使用的分布式检查点格式。
- **类型**: 字符串。
- **可选值**:
  - `zarr`
  - `torch_dist`
- **默认值**: `torch_dist`

#### `--ckpt-fully-parallel-save`
- **描述**: 已废弃：参见 `--no-ckpt-fully-parallel-save`。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--no-ckpt-fully-parallel-save`
- **描述**: 禁用在 DP 上完全并行保存分布式检查点。根据检查点格式，可能减少检查点中的文件数量。使分布式优化器检查点无法重新分片。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--async-save`
- **描述**: 应用异步检查点保存。当前仅与 `torch_dist` 分布式检查点格式兼容。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--ckpt-fully-parallel-load`
- **描述**: 在 DP 上完全并行加载分布式检查点。
- **类型**: 布尔值（使用 `store_true` 表表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--ckpt-assume-constant-structure`
- **描述**: 如果在*单个训练作业*中模型和优化器状态字典结构是恒定的，它允许不同的检查点性能优化。
- **类型**: 布尔值（使用 `store_true` 表表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--dist-ckpt-strictness`
- **描述**: 确定在检查点加载期间如何处理键不匹配。查看 StrictHandling 文档了解标志含义。注意：此标志仅控制从存储加载分布式检查点，不加载状态字典到模型。
- **类型**: 字符串。
- **可选值**:
  - [从 StrictHandling 枚举值中生成]
- **默认值**: `assume_ok_unexpected`

### 参数组：mixed precision

#### `--fp16`
- **描述**: 以 fp16 模式运行模型。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--bf16`
- **描述**: 以 bfloat16 模式运行模型。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--loss-scale`
- **描述**: 静态损失缩放，2 的正幂值可以改善 fp16 的收敛性。如果为 `None`，则使用动态损失缩放。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--initial-loss-scale`
- **描述**: 动态损失缩放的初始损失缩放值。
- **类型**: 浮点数。
- **默认值**: `4294967296` (2^32)

#### `--min-loss-scale`
- **描述**: 动态损失缩放的最小损失缩放值。
- **类型**: 浮点数。
- **默认值**: `1.0`

#### `--loss-scale-window`
- **描述**: 调整动态缩放比例的窗口大小。
- **类型**: 浮点数。
- **默认值**: `1000`

#### `--hysteresis`
- **描述**: 动态损失缩放的滞后。
- **类型**: 整数。
- **默认值**: `2`

#### `--fp32-residual-connection`
- **描述**: 将残差连接移至 fp32。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--apply-query-key-layer-scaling`
- **描述**: 通过层号对 Q * K^T 进行缩放（1 / 层号）。对 fp16 训练有帮助。同时将 `attention_softmax_in_fp32` 设置为 `True`。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--attention-softmax-in-fp32`
- **描述**: 在 fp32 中执行注意力遮罩和 softmax。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--accumulate-allreduce-grads-in-fp32`
- **描述**: 在 fp32 中进行梯度累积和全部减少。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--fp16-lm-cross-entropy`
- **描述**: 将语言模型头的未缩减交叉熵损失计算移至 fp16。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

### 参数组：distributed

#### `--tensor-model-parallel-size`
- **描述**: 张量模型并行的程度。
- **类型**: 整数。
- **默认值**: `1`

#### `--pipeline-model-parallel-size`
- **描述**: 流水线模型并行的程度。
- **类型**: 整数。
- **默认值**: `1`

#### `--encoder-pipeline-model-parallel-size`
- **描述**: 编码器中流水线模型并行的程度。
- **类型**: 整数。
- **默认值**: `None`

#### `--pipeline-model-parallel-split-rank`
- **描述**: 编码器和解码器应该分开的等级。已废弃；改用 `--encoder-pipeline-model-parallel-size`。
- **类型**: 整数。
- **默认值**: `None`

#### `--model-parallel-size`
- **描述**: 旧的模型并行参数，不建议使用。请使用 `--tensor-model-parallel-size`。
- **类型**: 整数。
- **默认值**: `None`

#### `--num-layers-per-virtual-pipeline-stage`
- **描述**: 每个虚拟流水线阶段的层数。
- **类型**: 整数。
- **默认值**: `None`

#### `--no-overlap-p2p-communication`
- **描述**: 不与前向和后向块重叠流水线并行通信。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--distributed-backend`
- **描述**: 用于分布式训练的后端。
- **类型**: 字符串。
- **可选值**:
  - `nccl`
  - `gloo`
- **默认值**: `nccl`

#### `--distributed-timeout-minutes`
- **描述**: `torch.distributed` 的超时时间（分钟）。
- **类型**: 整数。
- **默认值**: `10`

#### `--overlap-grad-reduce`
- **描述**: 如果设置，重叠 DDP 梯度减少。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--defer-embedding-wgrad-compute`
- **描述**: 如果设置，延迟词汇投影线性层权重梯度的计算至流水线冲刷。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--wgrad-deferral-limit`
- **描述**: 词汇投影权重梯度计算的延迟微批次数，默认为 0，意味着所有微批次都被延迟。
- **类型**: 整数。
- **默认值**: `0`

#### `--no-delay-grad-reduce`
- **描述**: 如果未设置，在除第一流水线阶段外的所有阶段延迟/同步梯度减少。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--ddp-bucket-size`
- **描述**: 数据并行通信的桶大小。
- **类型**: 整数。
- **默认值**: `None`

#### `--ddp-average-in-collective`
- **描述**: 如果设置，在数据并行通信集合中直接平均。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--overlap-param-gather`
- **描述**: 如果设置，在分布式优化器中重叠参数 all-gather。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--delay-param-gather`
- **描述**: 如果设置，在除第一流水线阶段外的所有阶段延迟/同步参数 all-gathers。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--no-scatter-gather-tensors-in-pipeline`
- **描述**: 如果未设置，使用 scatter/gather 优化流水线中张量的通信。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--use-ring-exchange-p2p`
- **描述**: 如果设置，使用自定义构建的环交换进行 p2p 通信。注意，此选项需要支持环交换 p2p 的自定义构建镜像。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--local_rank`
- **描述**: 从分布式启动器传递的本地等级。
- **类型**: 整数。
- **默认值**: `None`

#### `--lazy-mpu-init`
- **描述**: 如果设置为 True，则 `initialize_megatron()` 跳过 DDP 初始化并返回完成它的函数。也打开 `--use-cpu-initialization` 标志。这适用于外部 DDP 管理器。
- **类型**: 布尔值。
- **默认值**: `False`

#### `--standalone-embedding-stage`
- **描述**: 如果设置，*输入* 嵌入层放置在其自己的流水线阶段，不包括任何变换层。（对于 T5，此标志当前仅影响编码器嵌入。）
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--use-distributed-optimizer`
- **描述**: 使用分布式优化器。
- **类型**: 布尔值（使用 `store_true` 表表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--context-parallel-size`
- **描述**: 上下文并行的程度。
- **类型**: 整数。
- **默认值**: `1`

#### `--nccl-communicator-config-path`
- **描述**: NCCL 通信器配置的 YAML 文件路径。可以通过设置 `min_ctas`、`max_ctas` 和 `cga_cluster_size` 配置每个通信器的最小/最大线程组和线程组集群大小。
- **类型**: 字符串。
- **默认值**: `None`

#### `--use-tp-pp-dp-mapping`
- **描述**: 如果设置，分布式等级初始化顺序从 tp-dp-pp 更改为 tp-pp-dp。确保在启用此选项时不使用 EP 和 CP。
- **类型**: 布尔值（使用 `store_true` 表表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

### 参数组：validation

#### `--eval-iters`
- **描述**: 运行评估的迭代次数。
- **类型**: 整数。
- **默认值**: `100`

#### `--eval-interval`
- **描述**: 在验证集上运行评估的间隔。
- **类型**: 整数。
- **默认值**: `1000`

#### `--test-mode`
- **描述**: 运行实验时并行运行所有实时测试。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--skip-train`
- **描述**: 如果设置，跳过训练循环，可选地进行验证/测试评估并退出。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

### 参数组：data and dataloader

#### `--data-path`
- **描述**: 训练、验证和测试数据集的权重和前缀列表，根据 `--split` 进行分割。接受的格式包括：(1) 单个前缀，(2) 权重前缀对列表，例如 weight1 prefix1 weight2 prefix2，(3) 前缀列表，例如 prefix1 prefix2。对于 (3)，权重从贡献数据集的长度中推断。
- **类型**: 字符串列表。
- **默认值**: `None`

#### `--split`
- **描述**: 训练、验证和测试分割的逗号分隔比例列表。例如，`90,5,5` 表示 90% 的数据用于训练，5% 用于验证，5% 用于测试。
- **类型**: 字符串。
- **默认值**: `None`

#### `--train-data-path`
- **描述**: 独立训练数据集的权重和前缀列表。遵循与 `--data-path` 相同的模式规则。
- **类型**: 字符串列表。
- **默认值**: `None`

#### `--valid-data-path`
- **描述**: 独立验证数据集的权重和前缀列表。遵循与 `--data-path` 相同的模式规则。
- **类型**: 字符串列表。
- **默认值**: `None`

#### `--test-data-path`
- **描述**: 独立测试数据集的权重和前缀列表。遵循与 `--data-path` 相同的模式规则。
- **类型**: 字符串列表。
- **默认值**: `None`

#### `--data-cache-path`
- **描述**: 缓存索引文件的目录路径。
- **类型**: 字符串。
- **默认值**: `None`

#### `--no-mmap-bin-files`
- **描述**: 禁用 .bin 文件的 mmap-ing。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--mock-data`
- **描述**: 跳过数据加载和验证，选择在可用时人工生成模拟数据。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--vocab-size`
- **描述**: 在 EOD 或填充前的词汇量大小。
- **类型**: 整数。
- **默认值**: `None`

#### `--vocab-file`
- **描述**: 词汇表文件的路径。
- **类型**: 字符串。
- **默认值**: `None`

#### `--merge-file`
- **描述**: BPE合并文件的路径。
- **类型**: 字符串。
- **默认值**: `None`

#### `--vocab-extra-ids`
- **描述**: 额外词汇量的数量。在T5模型中用于跨度掩码。
- **类型**: 整数。
- **默认值**: `0`

#### `--seq-length`
- **描述**: 处理的最大序列长度。
- **类型**: 整数。
- **默认值**: `None`

#### `--encoder-seq-length`
- **描述**: 处理的最大编码器序列长度。此值应与 `--seq-length` 互斥。
- **类型**: 整数。
- **默认值**: `None`

#### `--decoder-seq-length`
- **描述**: 处理的最大解码器序列长度。
- **类型**: 整数。
- **默认值**: `None`

#### `--retriever-seq-length`
- **描述**: 检索器模型的最大序列长度。
- **类型**: 整数。
- **默认值**: `256`

#### `--sample-rate`
- **描述**: 训练数据的采样率。应为 0 < sample_rate < 1。
- **类型**: 浮点数。
- **默认值**: `1.0`

#### `--mask-prob`
- **描述**: 用掩码替换令牌的概率。
- **类型**: 浮点数。
- **默认值**: `0.15`

#### `--short-seq-prob`
- **描述**: 生成短序列的概率。
- **类型**: 浮点数。
- **默认值**: `0.1`

#### `--num-workers`
- **描述**: DataLoader的工作线程数。
- **类型**: 整数。
- **默认值**: `2`

#### `--tokenizer-type`
- **描述**: 使用的分词器类型。
- **类型**: 字符串。
- **可选值**:
  - `BertWordPieceLowerCase`
  - `BertWordPieceCase`
  - `GPT2BPETokenizer`
  - `SentencePieceTokenizer`
  - `GPTSentencePieceTokenizer`
  - `HuggingFaceTokenizer`
  - `Llama2Tokenizer`
  - `Llama3Tokenizer`
  - `MistralTokenizer`
  - `TikTokenizer`
  - `NullTokenizer`
- **默认值**: `None`

#### `--tokenizer-model`
- **描述**: Sentencepiece 分词器模型。
- **类型**: 字符串。
- **默认值**: `None`

#### `--tiktoken-pattern`
- **描述**: 使用的 tiktoken 模式。选项: [v1, v2]。
- **类型**: 字符串。
- **默认值**: `None`

#### `--tiktoken-num-special-tokens`
- **描述**: tiktoken 分词器中特殊令牌的数量。
- **类型**: 整数。
- **默认值**: `1000`

#### `--tiktoken-special-tokens`
- **描述**: tiktoken 特殊令牌的列表，需要包含 ["<unk>", "<s>", "</s>"]。
- **类型**: 字符串列表。
- **默认值**: `None`

#### `--reset-position-ids`
- **描述**: 在文档结束令牌后重置位置 ID。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--reset-attention-mask`
- **描述**: 在文档结束令牌后重置自注意力掩码。
- **类型**: 布尔值（使用 `store_true` 表表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--eod-mask-loss`
- **描述**: 掩盖文档结束令牌的损失。
- **类型**: 布尔值（使用 `store_true` 表表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--no-create-attention-mask-in-dataloader`
- **描述**: 如果设置，DataLoader中不创建注意力掩码。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--num-dataset-builder-threads`
- **描述**: 每个等级用于数据集构建器的并行线程数。
- **类型**: 整数。
- **默认值**: `1`

#### `--s3-cache-path`
- **描述**: 使用 s3 DataLoader 时缓存索引文件的路径。
- **类型**: 字符串。
- **默认值**: `None`

### 参数组：autoresume

#### `--adlr-autoresume`
- **描述**: 在 adlr 集群上启用自动恢复。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--adlr-autoresume-interval`
- **描述**: 检查自动恢复终止信号的间隔。
- **类型**: 整数。
- **默认值**: `1000`

### 参数组：biencoder

#### `--ict-head-size`
- **描述**: 在 ICT 和 REALM 中使用的块嵌入的大小（论文默认值：128）。
- **类型**: 整数。
- **默认值**: `None`

#### `--biencoder-projection-dim`
- **描述**: 双编码器中使用的投影头的大小（论文默认值：128）。
- **类型**: 整数。
- **默认值**: `0`

#### `--biencoder-shared-query-context-model`
- **描述**: 是否共享查询和上下文模型的参数。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--ict-load`
- **描述**: 包含 ICTBertModel 检查点的目录。
- **类型**: 字符串。
- **默认值**: `None`

#### `--bert-load`
- **描述**: 包含 BertModel 检查点的目录（开始 ICT 和 REALM 所需）。
- **类型**: 字符串。
- **默认值**: `None`

#### `--titles-data-path`
- **描述**: 用于 ICT 的标题数据集的路径。
- **类型**: 字符串。
- **默认值**: `None`

#### `--query-in-block-prob`
- **描述**: 在 ICT 数据集中保持查询在块中的概率。
- **类型**: 浮点数。
- **默认值**: `0.1`

#### `--use-one-sent-docs`
- **描述**: 是否在 ICT 中使用单句文档。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--evidence-data-path`
- **描述**: DPR 论文中的维基百科证据数据的路径。
- **类型**: 字符串。
- **默认值**: `None`

#### `--retriever-report-topk-accuracies`
- **描述**: 报告哪些 top-k 准确率（例如 '1 5 20'）。
- **类型**: 整数列表。
- **默认值**: `[]`

#### `--retriever-score-scaling`
- **描述**: 是否按隐藏大小的平方根的倒数缩放检索器得分。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--block-data-path`
- **描述**: 保存/加载 BlockData 的位置。
- **类型**: 字符串。
- **默认值**: `None`

#### `--embedding-path`
- **描述**: 保存/加载开放检索嵌入数据的位置。
- **类型**: 字符串。
- **默认值**: `None`

#### `--indexer-batch-size`
- **描述**: 进行索引工作时使用的批量大小。
- **类型**: 整数。
- **默认值**: `128`

#### `--indexer-log-interval`
- **描述**: 索引器应在多少批次后报告进度。
- **类型**: 整数。
- **默认值**: `1000`

### 参数组：vision

#### `--num-classes`
- **描述**: 视觉分类任务中的类别数量。
- **类型**: 整数。
- **默认值**: `1000`

#### `--img-h`
- **描述**: 视觉分类任务的图像高度。
- **类型**: 整数。
- **默认值**: `224`

#### `--img-w`
- **描述**: 视觉分类任务的图像宽度。
- **类型**: 整数。
- **默认值**: `224`

#### `--num-channels`
- **描述**: 输入图像数据中的通道数。
- **类型**: 整数。
- **默认值**: `3`

#### `--patch-dim`
- **描述**: 补丁尺寸。
- **类型**: 整数。
- **默认值**: `16`

#### `--classes-fraction`
- **描述**: 使用类别的一部分进行训练。
- **类型**: 浮点数。
- **默认值**: `1.0`

#### `--data-per-class-fraction`
- **描述**: 每个类别使用数据的一部分进行训练。
- **类型**: 浮点数。
- **默认值**: `1.0`

#### `--no-data-sharding`
- **描述**: 禁用数据分片。
- **类型**: 布尔值（使用 `store_false` 表示如果指定了该参数，则设置为假）。
- **默认值**: `True`

#### `--head-lr-mult`
- **描述**: 微调过程中头部的学习率倍数。
- **类型**: 浮点数。
- **默认值**: `1.0`

#### `--vision-pretraining`
- **描述**: 指示视觉预训练的标志。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--vision-pretraining-type`
- **描述**: 预训练目标。
- **类型**: 字符串。
- **可选值**:
  - `classify`
  - `inpaint`
  - `dino`
- **默认值**: `classify`

#### `--vision-backbone-type`
- **描述**: 背景类型。
- **类型**: 字符串。
- **可选值**:
  - `vit`
  - `mit`
  - `swin`
- **默认值**: `vit`

#### `--swin-backbone-type`
- **描述**: Swin 背景类型。
- **类型**: 字符串。
- **可选值**:
  - `tiny`
  - `base`
  - `h3`
- **默认值**: `tiny`

#### `--mask-type`
- **描述**: 掩码类型。
- **类型**: 字符串。
- **可选值**:
  - `random`
  - `row`
- **默认值**: `random`

#### `--mask-factor`
- **描述**: 掩码大小的缩放参数。
- **类型**: 浮点数。
- **默认值**: `1.0`

#### `--iter-per-epoch`
- **描述**: 每个周期的迭代次数。
- **类型**: 整数。
- **默认值**: `1250`

#### `--dino-local-img-size`
- **描述**: DINO任务的局部图像大小。
- **类型**: 整数。
- **默认值**: `96`

#### `--dino-local-crops-number`
- **描述**: 局部裁剪的数量。
- **类型**: 整数。
- **默认值**: `10`

#### `--dino-head-hidden-size`
- **描述**: DINO头部的隐藏层尺寸。
- **类型**: 整数。
- **默认值**: `2048`

#### `--dino-bottleneck-size`
- **描述**: DINO头部的瓶颈尺寸。
- **类型**: 整数。
- **默认值**: `256`

#### `--dino-freeze-last-layer`
- **描述**: 冻结最后一层的权重。
- **类型**: 浮点数。
- **默认值**: `1`

#### `--dino-norm-last-layer`
- **描述**: 在最后一层禁用标准化。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--dino-warmup-teacher-temp`
- **描述**: DINO教师温度的预热。
- **类型**: 浮点数。
- **默认值**: `0.04`

#### `--dino-teacher-temp`
- **描述**: DINO的教师温度。
- **类型**: 浮点数。
- **默认值**: `0.07`

#### `--dino-warmup-teacher-temp-epochs`
- **描述**: DINO教师温度预热的周期数。
- **类型**: 整数。
- **默认值**: `30`

#### `--qk-layernorm`
- **描述**: 是否对注意力嵌入的 q 和 k 进行层归一化。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

### 参数组：moe

#### `--expert-model-parallel-size`
- **描述**: 专家模型并行的程度。
- **类型**: 整数。
- **默认值**: `1`

#### `--num-experts`
- **描述**: MoE中的专家数量（`None` 表示没有 MoE）。
- **类型**: 整数。
- **默认值**: `None`

#### `--moe-router-load-balancing-type`
- **描述**: 决定路由器的负载平衡策略。"aux_loss" 对应于 GShard 和 SwitchTransformer 中使用的负载平衡损失，"sinkhorn" 对应于 S-BASE 中使用的平衡算法，"none" 表示没有负载平衡。默认为 "aux_loss"。
- **类型**: 字符串。
- **可选值**:
  - `aux_loss`
  - `sinkhorn`
  - `none`
- **默认值**: `aux_loss`

#### `--moe-router-topk`
- **描述**: 每个令牌路由到的专家数量。默认为 2。
- **类型**: 整数。
- **默认值**: `2`

#### `--moe-grouped-gemm`
- **描述**: 当每个等级有多个专家时，将多个本地（可能较小的）gemms 压缩到一个内核启动中，通过利用自 CUTLASS 2.8 以来引入的分组 GEMM 功能来提高利用率和性能。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--moe-aux-loss-coeff`
- **描述**: 辅助损失的缩放系数：建议起始值为 1e-2。
- **类型**: 浮点数。
- **默认值**: `0.0`

#### `--moe-z-loss-coeff`
- **描述**: z 损失的缩放系数：建议起始值为 1e-3。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--moe-input-jitter-eps`
- **描述**: 通过使用指定的 epsilon 值添加抖动，向输入张量添加噪声。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--moe-token-dispatcher-type`
- **描述**: 选择令牌调度器的类型。
- **类型**: 字符串。
- **可选值**:
  - `allgather`
  - `alltoall`
- **默认值**: `allgather`

#### `--moe-per-layer-logging`
- **描述**: 启用 MoE 的逐层日志记录，目前支持辅助损失和 z 损失。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--moe-expert-capacity-factor`
- **描述**: 每个专家的容量因子，`None` 表示不会丢弃任何令牌。
- **类型**: 浮点数。
- **默认值**: `None`

#### `--moe-pad-expert-input-to-capacity`
- **描述**: 为每个专家的输入填充以匹配专家容量长度，仅在设置了 `--moe-expert-capacity-factor` 后生效。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--moe-token-drop-policy`
- **描述**: 丢弃令牌的策略。可以是 "probs" 或 "position"。如果是 "probs"，将丢弃概率最低的令牌。如果是 "position"，将丢弃每批次末尾的令牌。
- **类型**: 字符串。
- **可选值**:
  - `probs`
  - `position`
- **默认值**: `probs`

#### `--moe-layer-recompute`
- **描述**: 启用 moe_layer 的检查点，应在内存不足时使用。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

#### `--moe-extended-tp`
- **描述**: 专家并行的替代方案，所有专家都分布在 TPXEP 领域。
- **类型**: 布尔值（使用 `store_true` 表示如果指定了该参数，则设置为真）。
- **默认值**: `False`

### 参数组：experimental

#### `--spec`
- **描述**: 指定返回规范的 `<module_location function_name>` 对，用于根据用例定制模型、变压器块或变压器层。若使用本地规范，请指定 `local` 作为参数。更多细节参见模型类、`transformer_block.py` 或 `transformer_layer.py`。
- **类型**: 字符串列表。
- **默认值**: `None`

#### `--hybrid-attention-ratio`
- **描述**: 总层数中注意力层的比例，范围为 [0.0, 1.0]。
- **类型**: 浮点数。
- **默认值**: `0.0`

#### `--hybrid-mlp-ratio`
- **描述**: 总层数中 mlp 层的比例，范围为 [0.0, 1.0]。
- **类型**: 浮点数。
- **默认值**: `0.0`

#### `--hybrid-override-pattern`
- **描述**: 强制使用特定的混合层模式。如果任何混合比例参数提供了大于 0.0 的值，则覆盖模式中每种类型层的数量必须与被覆盖模式中的数量匹配。
- **类型**: 字符串。
- **默认值**: `None`

#### `--yaml-cfg`
- **描述**: 配置文件以添加额外参数。
- **类型**: 字符串。
- **默认值**: `None`

## config.yaml内部参数说明
如果你使用[config.yaml形式运行训练脚本](#使用configyaml运行模型训练脚本)，以下是配置参数的详细说明：

### 参数组：RWKV arguments

#### `check_against_pretrained`
- **描述**: 此属性用于决定是否将当前模型的输出与预训练模型的输出进行比较。当设置为True时，当前模型在前向传播中将执行额外的步骤，调用ground_truth_model_closure方法以获取预训练模型的实例，并使用钩子函数（hook）来捕获和比较两个模型的中间层输出。这通常用于调试或确保模型修改后的行为仍与预训练模型一致。
- **类型**: 布尔值。
- **默认值**: `False`

#### `load_pretrained_rwkv_source_type`
- **描述**: 此属性用于指定预训练模型的来源类型，决定了如何加载预训练模型到当前模型中。它支持从不同来源加载模型，如从 Hugging Face 平台（标记为 `mobius-huggingface`）或直接从 RWKV 语言模型文件（标记为 `rwkv-lm`）。根据此属性的值，加载函数将调用不同的方法来实现模型的初始化，确保与预训练源的兼容性。
- **类型**: 字符串。
- **默认值**: 无默认值，必须显式指定来源类型。

#### `load_pretrained_rwkv`
- **描述**: 此参数用于指定要加载的预训练模型的标识或路径。当设置此参数时，系统会根据提供的路径或标识从相应来源加载预训练模型。这通常在初始化模型或迁移学习场景中使用，允许用户从现有的预训练模型开始，而不是从头开始训练。
- **类型**: 字符串。
- **默认值**: `None`，表示默认情况下不从预训练模型加载。只有在明确指定预训练模型的标识或路径时才进行加载。

#### `debug_dump`
- **描述**: 此参数用于启动调试信息的输出。当`debug_dump`被设置为`True`时，代码将执行一系列调试操作，包括打印词汇表大小，同步分布式训练的不同节点，并按节点打印每个参数的名称和形状。此功能主要用于分布式训练环境中，以确保所有节点的模型参数同步一致，以及诊断可能的分布式配置错误。
- **类型**: 布尔值。
- **默认值**: `False`，表示默认不进行调试信息的输出，只有在显式设置为`True`时才进行。

#### `auto_train_iters`
- **描述**: 自动根据config里面的数据集大小计算只训练一个epoch的迭代次数，并更新配置
- **类型**: 布尔值。
- **默认值**: `False`

## transformer_config
### 参数组：Model parallelism

#### `num_layers`
- **描述**: 转换器块中的变换器层数。
- **类型**: 整数。
- **默认值**: `0`

#### `hidden_size`
- **描述**: 转换器隐藏尺寸。
- **类型**: 整数。
- **默认值**: `0`

#### `num_attention_heads`
- **描述**: 转换器注意力头数。
- **类型**: 整数。
- **默认值**: `0`

#### `num_query_groups`
- **描述**: 分组查询注意力的查询组数。如果为 `None`，则使用正常注意力。
- **类型**: 整数。
- **默认值**: `None`

#### `ffn_hidden_size`
- **描述**: 转换器前馈网络隐藏尺寸。如果未提供，该值设置为 `4*hidden_size`。
- **类型**: 整数。
- **默认值**: `None`

#### `kv_channels`
- **描述**: 多头注意力中的投影权重维数。如果未提供，该值设置为 `hidden_size // num_attention_heads`。
- **类型**: 整数。
- **默认值**: `None`

#### `hidden_dropout`
- **描述**: 转换器隐藏状态的失活概率。
- **类型**: 浮点数。
- **默认值**: `0.1`

#### `attention_dropout`
- **描述**: 注意力之后的失活概率。
- **类型**: 浮点数。
- **默认值**: `0.1`

#### `fp32_residual_connection`
- **描述**: 如果为真，将残差连接移至 fp32。
- **类型**: 布尔值。
- **默认值**: `False`

#### `apply_residual_connection_post_layernorm`
- **描述**: 如果为真，使用原始BERT残差连接顺序。
- **类型**: 布尔值。
- **默认值**: `False`

#### `layernorm_epsilon`
- **描述**: 任何LayerNorm操作的Epsilon值。
- **类型**: 浮点数。
- **默认值**: `1e-5`

#### `layernorm_zero_centered_gamma`
- **描述**: 如果设为真，调整LayerNorm以使gamma值围绕0中心。这改善了数值稳定性。
- **类型**: 布尔值。
- **默认值**: `False`

#### `add_bias_linear`
- **描述**: 在所有线性层中包括一个偏差项（QKV投影，注意力核心之后，以及MLP层中的两个）。
- **类型**: 布尔值。
- **默认值**: `True`

#### `add_qkv_bias`
- **描述**: 仅为QKV投影添加一个偏差项。
- **类型**: 布尔值。
- **默认值**: `False`

#### `gated_linear_unit`
- **描述**: 在MLP的第一个线性层中使用门控线性单元。
- **类型**: 布尔值。
- **默认值**: `False`

#### `activation_func`
- **描述**: 在MLP中使用的非线性激活函数。
- **类型**: 可调用。
- **默认值**: `F.gelu`

#### `activation_func_fp8_input_store`
- **描述**: 在反向传播中保存MLP激活函数的输入在FP8中以节省内存。存储的输入在反向传播计算前转换回原始精度。
- **类型**: 布尔值。
- **默认值**: `False`

#### `num_moe_experts`
- **描述**: 用于MoE层的专家数目。设置后，用MoE层替换MLP。设为 `None` 则不使用MoE。
- **类型**: 整数。
- **默认值**: `None`

#### `rotary_interleaved`
- **描述**: 如果为真，则轮换偶数和奇数维度的对（RoFormer风格），假则轮换前半部分和后半部分的对（LLaMa风格）。默认为假。
- **类型**: 布尔值。
- **默认值**: `False`

#### `window_size`
- **描述**: 如果非空，则使用滑动窗口注意力。窗口的尺寸由元组内的数字指定；-1 是特殊值，意味着“无限窗口大小”。
- **类型**: 元组。
- **默认值**: `None`

#### `normalization`
- **描述**: 用于归一化层的标准化方式，有效选项是 `LayerNorm` 和 `RMSNorm`。
- **类型**: 字符串。
- **默认值**: `"LayerNorm"`

#### `qk_layernorm`
- **描述**: 是否对查询和键嵌入应用LayerNorm。
- **类型**: 布尔值。
- **默认值**: `False`

#### `test_mode`
- **描述**: 是否运行实时测试。
- **类型**: 布尔值。
- **默认值**: `False`

#### `calculate_per_token_loss`
- **描述**: 交叉熵损失是根据实际非填充标记的数量计算还是假设所有标记都是非填充的，默认行为。
- **类型**: 布尔值。
- **默认值**: `False`

### 参数组：initialization

#### `init_method`
- **描述**: 权重初始化方法。请注意，偏差总是设置为零。应为一个函数，该函数接受一个张量并初始化它。如果为 `None`，将设置为 `megatron.core.utils.init_method_normal(init_method_std)`，这是 torch nn init normal，均值为 `0.0`，标准差为 `init_method_std`。
- **类型**: 可调用。
- **默认值**: `None`

#### `output_layer_init_method`
- **描述**: 初始化注意力和MLP块的输出层权重的方法。如果为 `None`，将设置为 `megatron.core.utils.scaled_init_method_normal(init_method_std)`，这是 torch nn init normal，均值为 `0.0`，标准差为 `init_method_std / math.sqrt(2.0 * num_layers)`。
- **类型**: 可调用。
- **默认值**: `None`

#### `init_method_std`
- **描述**: 默认初始化方法的零均值正态分布的标准差，如果提供了 `init_method` 和 `output_layer_init_method`，则不使用。
- **类型**: 浮点数。
- **默认值**: `0.02`

### 参数组：mixed-precision

#### `apply_query_key_layer_scaling`
- **描述**: 如果为真，通过 `1 / layer-number` 缩放 `Q * K^T`。这在使用 fp16 训练时可以改善数值稳定性。
- **类型**: 布尔值。
- **默认值**: `False`

#### `attention_softmax_in_fp32`
- **描述**: 如果为真，执行注意力遮罩和 softmax 在 fp32 中。如果 `apply_query_key_layer_scaling` 为真，该项应为真。
- **类型**: 布尔值。
- **默认值**: `True`

### 参数组：fusion

#### `bias_activation_fusion`
- **描述**: 如果为真，当可能时合并偏差添加和激活函数。
- **类型**: 布尔值。
- **默认值**: `False`

#### `masked_softmax_fusion`
- **描述**: 如果为真，使用 softmax 融合。
- **类型**: 布尔值。
- **默认值**: `False`

#### `persist_layer_norm`
- **描述**: 如果为真，使用持久化融合层规范化内核。此内核只支持固定的隐藏尺寸集。
- **类型**: 布尔值。
- **默认值**: `False`

#### `memory_efficient_layer_norm`
- **描述**: 如果为真，并且使用本地层（不是来自 TransformerEngine），告诉 Apex 使用内存高效的融合层规范化内核。如果不使用 LayerNorm，则忽略此项。
- **类型**: 布尔值。
- **默认值**: `False`

#### `bias_dropout_fusion`
- **描述**: 如果为真，使用偏差失活融合。# TODO: 这应该是 bias_dropout_add_fusion?
- **类型**: 布尔值。
- **默认值**: `False`

#### `apply_rope_fusion`
- **描述**: 如果为真，使用融合 RoPE 内核。
- **类型**: 布尔值。
- **默认值**: `False`

### 参数组：activation recomputation

#### `recompute_granularity`
- **描述**: 决定使用哪种类型的激活重算。Megatron-core 支持选择性激活检查点，其中只有注意力的内存密集部分被检查点。这些内存密集型激活也较少计算密集，使得激活检查点对于大型语言模型（20B+）更为高效。有关更多详细信息，请参见《减少大型变换器模型中的激活重算》(https://arxiv.org/abs/2205.05198)。'full' 将检查点整个变换器层。如果为 `None`，则不执行重算，所有激活都将被保存。如果设置，必须是 'selective' 或 'full'。'selective' 总是使用所有层。
- **类型**: 字符串。
- **默认值**: `None`

#### `recompute_method`
- **描述**: 决定哪些变换器层将被重新计算。uniform 将均匀地分割变换器块中的变换器层数量，并在指定的粒度下重新计算每个分割块的输入激活。block 将仅为每个流水线阶段的一组设置的变换器层重新计算输入激活。流水线阶段中的其他层将不会重新计算任何激活。如果为 `None`，并且启用了重算，所有层都将进行重算。如果设置，必须是 'uniform' 或 'block'。
- **类型**: 字符串。
- **默认值**: `None`

#### `recompute_num_layers`
- **描述**: 当 `recompute_method` 为 uniform 时，`recompute_num_layers` 是每个均匀分割重算单元中的变换器层数。当 `recompute_method` 为 block 时，`recompute_num_layers` 是每个流水线阶段内要重算的变换器层数。对于 'selective' 激活检查点，必须为 `None`。
- **类型**: 整数。
- **默认值**: `None`

#### `distribute_saved_activations`
- **描述**: 如果为真，将重算的激活分布到模型并行组中。
- **类型**: 布尔值。
- **默认值**: `None`

### 参数组：fp8 related

#### `fp8`
- **描述**: 如果设置，通过 Transformer Engine 启用 FP8 精度。有两个预定义选项：(1) 'e4m3' 均匀使用 e4m3 对所有 FP8 张量，(2) 'hybrid' 对所有 FP8 激活和权重张量使用 e4m3，对所有 FP8 输出激活梯度张量使用 e5m2。
- **类型**: 字符串。
- **默认值**: `None`

#### `fp8_margin`
- **描述**: 缩放因子计算的余量。
- **类型**: 整数。
- **默认值**: `0`

#### `fp8_interval`
- **描述**: 控制多久重新计算一次缩放因子。
- **类型**: 整数。
- **默认值**: `1`

#### `fp8_amax_history_len`
- **描述**: 用于缩放因子计算的 amax 历史窗口长度。
- **类型**: 整数。
- **默认值**: `1`

#### `fp8_amax_compute_algo`
- **描述**: 选择缩放因子计算的 `amax` 值的算法。有两个预定义选项：`max` 选择历史窗口中最大的 `amax`，而 `most_recent` 总是选择最近看到的值。
- **类型**: 字符串。
- **默认值**: `"most_recent"`

#### `fp8_wgrad`
- **描述**: 当设置为 False 时，覆盖 FP8 配置选项，并在更高精度中进行 wgrad 计算。
- **类型**: 布尔值。
- **默认值**: `True`

#### `fp8_dot_product_attention`
- **描述**: 当设置为 True 时，使用 FP8 实现的点积注意力。
- **类型**: 布尔值。
- **默认值**: `False`

#### `fp8_multi_head_attention`
- **描述**: 当设置为 True 时，使用 FP8 实现的多头注意力。
- **类型**: 布尔值。
- **默认值**: `False`

### 参数组：MoE related

#### `moe_router_load_balancing_type`
- **描述**: 决定路由器的负载均衡策略。"aux_loss" 对应于 GShard 和 SwitchTransformer 中使用的负载均衡损失，"sinkhorn" 对应于 S-BASE 中使用的平衡算法，而 "none" 表示没有负载均衡。
- **类型**: 字符串。
- **默认值**: `"aux_loss"`

#### `moe_router_topk`
- **描述**: 每个令牌路由到的专家数量。
- **类型**: 整数。
- **默认值**: `2`

#### `moe_router_pre_softmax`
- **描述**: 启用 MoE 的预 softmax 路由，这意味着 softmax 在 top-k 选择之前完成。默认情况下，softmax 在 top-k 之后完成。
- **类型**: 布尔值。
- **默认值**: `False`

#### `moe_grouped_gemm`
- **描述**: 当每个等级有多个专家时，将多个本地（可能较小的）gemm 压缩到单个内核启动中，以通过利用自 CUTLASS 2.8 以来引入的 Grouped GEMM 功能来提高利用率和性能（https://github.com/fanshiqing/grouped_gemm）。
- **类型**: 布尔值。
- **默认值**: `False`

#### `moe_aux_loss_coeff`
- **描述**: aux 损失的缩放系数。推荐起始值为 1e-2。
- **类型**: 浮点数。
- **默认值**: `0`

#### `moe_z_loss_coeff`
- **描述**: z-loss 的缩放系数。推荐起始值为 1e-3。
- **类型**: 浮点数。
- **默认值**: `None`

#### `moe_input_jitter_eps`
- **描述**: 通过应用具有指定 epsilon 值的抖动来向输入张量添加噪声。
- **类型**: 浮点数。
- **默认值**: `None`

#### `moe_token_dropping`
- **描述**: 此功能涉及为每个专家选择性地删除和填充令牌以实现指定容量，类似于 GShard、Switch-Transformer 和 DeepSpeed-MoE。请注意，这目前不受支持，因此应保持为 False。
- **类型**: 布尔值。
- **默认值**: `False`

#### `moe_token_dispatcher_type`
- **描述**: 使用的令牌分发器类型。默认为 'allgather'。选项为 'allgather' 和 'alltoall'。
- **类型**: 字符串。
- **默认值**: `"allgather"`

#### `moe_per_layer_logging`
- **描述**: 启用 MoE 的逐层日志记录，目前支持辅助损失和 z 损失。
- **类型**: 布尔值。
- **默认值**: `False`

#### `moe_expert_capacity_factor`
- **描述**: 每个专家的容量因子，`None` 表示不会丢弃任何令牌。默认值为 `None`。
- **类型**: 浮点数。
- **默认值**: `None`

#### `moe_pad_expert_input_to_capacity`
- **描述**: 如果为真，为每个专家的输入填充以匹配专家容量长度，仅在设置了 `moe_expert_capacity_factor` 后有效。默认设置为 False。
- **类型**: 布尔值。
- **默认值**: `False`

#### `moe_token_drop_policy`
- **描述**: 丢弃令牌的策略。可以是 "probs" 或 "position"。如果是 "probs"，将丢弃概率最低的令牌。如果是 "position"，将丢弃每批次末尾的令牌。
- **类型**: 字符串。
- **默认值**: `'probs'`

#### `moe_layer_recompute`
- **描述**: 内存优化：检查点 moe_layer 以节省激活内存。
- **类型**: 布尔值。
- **默认值**: `False`

### 参数组：miscellaneous

#### `clone_scatter_output_in_embedding`
- **描述**: 当设置为真时，克隆在嵌入层中 scatter_to_sequence_parallel_region 的输出，以促进输入的垃圾回收。
- **类型**: 布尔值。
- **默认值**: `True`

#### `disable_parameter_transpose_cache`
- **描述**: 当设置为真时，不缓存后续迭代的参数转置。
- **类型**: 布尔值。
- **默认值**: `False`

#### `enable_cuda_graph`
- **描述**: 当设置为真时，将 TransformerLayer 块包装在 CUDA 图中。
- **类型**: 布尔值。
- **默认值**: `False`

#### `config_logger_dir`
- **描述**: 当非空时，将入口点配置转储到 config_logger_dir。
- **类型**: 字符串。
- **默认值**: `""`

## model_parallel_config
### 参数组：Model parallelism

#### `tensor_model_parallel_size`
- **描述**: 层内模型并行性。将张量在 GPU 等级间分割。
- **类型**: 整数。
- **默认值**: `1`

#### `pipeline_model_parallel_size`
- **描述**: 层间模型并行性。将变换器层在 GPU 等级间分割。
- **类型**: 整数。
- **默认值**: `1`

#### `virtual_pipeline_model_parallel_size`
- **描述**: 交错管道并行性用于通过减少管道泡沫来提高性能。将一个变换器块视为一系列较小的变换器（虚拟）块。每个管道模型并行等级的虚拟块数量是虚拟模型并行大小。有关更多详细信息，请参见《在 GPU 集群上使用 Megatron-LM 进行高效的大规模语言模型训练》：arxiv.org/pdf/2104.04473.pdf。
- **类型**: 整数（可选）。
- **默认值**: `None`

#### `sequence_parallel`
- **描述**: 通过并行化层规范和失活来使张量并行性对大型语言模型（20B+）更加内存高效。有关更多详细信息，请参见《减少大型变换器模型中的激活重算》(https://arxiv.org/abs/2205.05198)。
- **类型**: 布尔值。
- **默认值**: `False`

#### `context_parallel_size`
- **描述**: 沿序列维度在 GPU 等级间分割网络输入。
- **类型**: 整数。
- **默认值**: `1`

#### `expert_model_parallel_size`
- **描述**: 在子数据并行维度上分布 MoE 专家。
- **类型**: 整数。
- **默认值**: `1`

#### `moe_extended_tp`
- **描述**: 专家并行性的替代并行策略。与其在 expert_model_parallel_size 上分布专家，不如将每个专家在扩展的张量并行域（tensor_model_parallel_size * expert_model_parallel_size）上分片。它避免了 MOE 训练的负载平衡问题。
- **类型**: 布尔值。
- **默认值**: `False`

### 参数组：Initialization

#### `perform_initialization`
- **描述**: 如果为真，权重将被初始化。当您知道将要从检查点加载值时，此选项可能很有用。
- **类型**: 布尔值。
- **默认值**: `True`

#### `use_cpu_initialization`
- **描述**: 当设置为 False 时，我们直接在 GPU 上初始化权重。CPU 初始化与张量模型并行性无关，但 GPU 初始化则不同。从 CPU 到 GPU 转移权重可能需要大型模型的大量时间。
- **类型**: 布尔值。
- **默认值**: `False`

### 参数组：Training

#### `fp16`
- **描述**: 如果为真，使用 fp16 混合精度训练。
- **类型**: 布尔值。
- **默认值**: `False`

#### `bf16`
- **描述**: 如果为真，使用 bf16 混合精度训练。
- **类型**: 布尔值。
- **默认值**: `False`

#### `params_dtype`
- **描述**: 初始化权重时使用的数据类型。
- **类型**: torch.dtype。
- **默认值**: `torch.float32`

#### `timers`
- **描述**: 计时器对象，用于调用各种计时功能。请参阅 megatron.core.timers.Timers。
- **类型**: 可调用。
- **默认值**: `None`

#### `finalize_model_grads_func`
- **描述**: 在所有工作节点上完成梯度的函数。可能包括确保梯度在数据并行性、管道并行性和序列并行性维度上进行全约减。
- **类型**: 可调用。
- **默认值**: `None`

#### `grad_scale_func`
- **描述**: 如果使用损失缩放，此函数应取损失并返回缩放后的损失。如果为 `None`，则不在损失上调用任何函数。
- **类型**: 可调用。
- **默认值**: `None`

#### `no_sync_func`
- **描述**: 创建一个抑制异步数据并行通信的上下文的函数。如果模型是 core.distributed.DistributedDataParallel 的实例，默认使用 core.distributed.DistributedDataParallel.no_sync。
- **类型**: 可调用。
- **默认值**: `None`

#### `grad_sync_func`
- **描述**: 启动异步梯度缩减的函数（例如，分布式优化器梯度 reduce-scatters）。该函数应取一个参数：其梯度需要同步的参数的可迭代对象。
- **类型**: 可调用。
- **默认值**: `None`

#### `param_sync_func`
- **描述**: 启动异步参数同步的函数（例如，分布式优化器参数 all-gathers）。该函数应取一个参数：需要同步的参数的可迭代对象。
- **类型**: 可调用。
- **默认值**: `None`

#### `deterministic_mode`
- **描述**: 如果为真，将选择具有确定性执行的代码。这通常意味着执行速度更慢，但有助于调试和测试。
- **类型**: 布尔值。
- **默认值**: `False`

#### `enable_autocast`
- **描述**: 如果为真，则在 torch.autocast 上下文中运行前向步骤函数。
- **类型**: 布尔值。
- **默认值**: `False`

#### `autocast_dtype`
- **描述**: 启用时传递给 torch.amp.autocast 的数据类型。如果为 `None`，则设置为 pipeline_dtype。
- **类型**: torch.dtype。
- **默认值**: `None`

#### `num_microbatches_with_partial_activation_checkpoints`
- **描述**: 如果为整数，设置不会检查点并重新计算所有层的微批次数量。在最大未完成微批次窗口内的其他微批次将重新计算所有层（全重新计算或选择性重新计算）。如果为 `None`，检查点和重新计算将由前向步骤函数决定。
- **类型**: 整数（可选）。
- **默认值**: `None`

### 参数组：Optimizations

#### `gradient_accumulation_fusion`
- **描述**: 如果为真，将权重梯度累积与 GEMMs 融合。需要自定义 CUDA 扩展 fused_weight_gradient_mlp_cuda 模块。要使用 gradient_accumulation_fusion，必须安装带有 --cpp_ext 和 --cuda_ext 的 APEX。例如：`pip install --global-option="--cpp_ext" --global-option="--cuda_ext"`。请注意，扩展要求 CUDA>=11。否则，必须关闭梯度累积融合。
- **类型**: 布尔值。
- **默认值**: `False`

#### `async_tensor_model_parallel_allreduce`
- **描述**: 已弃用。此标志被忽略。
- **类型**: 布尔值。
- **默认值**: `False`

#### `use_te_rng_tracker`
- **描述**: 如果为真，在 TransformerEngine 中使用 RNG 状态跟踪器（如果存在）。
- **类型**: 布尔值。
- **默认值**: `False`

#### `tp_comm_overlap`
- **描述**: 如果为真，允许在张量并行通信集合（如 AllGather/ReduceScatter）期间与 Linear 层执行重叠。在前向和反向传递期间尽可能进行 Linear 层的重叠。
- **类型**: 布尔值。
- **默认值**: `False`

#### `tp_comm_bulk_wgrad`
- **描述**: 如果为真，允许与 Bprop 激活梯度 GEMM 的 All-Gather 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `True`

#### `tp_comm_bulk_dgrad`
- **描述**: 如果为真，允许与 Bprop 权重梯度 GEMM 的 Reduce-Scatter 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `True`

#### `tp_comm_overlap_ag`
- **描述**: 如果为真，通过对 GEMM 和 All-Gather 进行流水线处理，允许与 GEMM 的 All-Gather 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `True`

#### `tp_comm_overlap_rs`
- **描述**: 如果为真，通过对 GEMM 和 Reduce-Scatter 进行流水线处理，允许与 GEMM 的 Reduce-Scatter 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `True`

#### `tp_comm_overlap_rs_dgrad`
- **描述**: 如果为真，通过对 GEMM 和 Reduce-Scatter 分割进行流水线处理，允许与 DGRAD GEMM 的 Reduce-Scatter 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `False`

#### `tp_comm_split_ag`
- **描述**: 已从 TransformerEngine v1.6.0 弃用。如果为真，通过对 GEMM 和 All-Gather 分割进行流水线处理，允许与 Fprop GEMM 的 All-Gather 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `True`

#### `tp_comm_atomic_ag`
- **描述**: 已从 TransformerEngine v1.6.0 弃用。如果为真，通过对 GEMM 和 All-Gather 同时进行原子操作，允许与 Fprop GEMM 的 All-Gather 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `False`

#### `tp_comm_split_rs`
- **描述**: 已从 TransformerEngine v1.6.0 弃用。如果为真，通过对 GEMM 和 Reduce-Scatter 分割进行流水线处理，允许与 Fprop GEMM 的 Reduce-Scatter 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `True`

#### `tp_comm_atomic_rs`
- **描述**: 已从 TransformerEngine v1.6.0 弃用。如果为真，通过对 GEMM 和 Reduce-Scatter 同时进行原子操作，允许与 Fprop GEMM 的 Reduce-Scatter 重叠。如果 tp_comm_overlap 为 False，则不考虑。
- **类型**: 布尔值。
- **默认值**: `False`

#### `cross_entropy_loss_fusion`
- **描述**: 如果启用，将使用融合的交叉熵实现。
- **类型**: 布尔值。
- **默认值**: `False`

#### `tp_comm_overlap_disable_qkv`
- **描述**: 如果为真，禁用 QKV 的 AllGather -> GEMM 重叠。
- **类型**: 布尔值。
- **默认值**: `False`

#### `tp_comm_overlap_disable_fc1`
- **描述**: 如果为真，禁用 MLP 的 FC1 层的 AllGather -> GEMM 重叠。
- **类型**: 布尔值。
- **默认值**: `False`

### 参数组：Pipeline Parallel

#### `pipeline_dtype`
- **描述**: 在点对点通信中使用的数据类型，通常是 `params_dtype`。
- **类型**: torch.dtype。
- **默认值**: `None`

#### `variable_seq_lengths`
- **描述**: 支持微批次间变化的序列长度。设置此项将在管道并行通信中传递张量大小，因此由于这种额外开销，只有在全局批次内微批次的序列长度有变化时才应设置。
- **类型**: 布尔值。
- **默认值**: `False`

#### `overlap_p2p_comm`
- **描述**: 当为 True 时，一些管道并行的点对点通信将与计算重叠。如果 `batch_p2p_comm` 为 True，则必须为 False。
- **类型**: 布尔值。
- **默认值**: `False`

#### `batch_p2p_comm`
- **描述**: 使用 `batch_isend_irecv` 而不是单独的 `isend/irecv` 调用。如果 `overlap_p2p_comm` 为 True，则必须为 False。
- **类型**: 布尔值。
- **默认值**: `True`

#### `batch_p2p_sync`
- **描述**: 使用 `batch_isend_irecv` 时，之后执行 `cuda.device.synchronize`，以解决旧版 PyTorch 中的一个错误。
- **类型**: 布尔值。
- **默认值**: `True`

#### `use_ring_exchange_p2p`
- **描述**: 使用自定义 `ring_exchange` 内核而不是 `torch.distributed.batch_isend_irecv()`。需要定制构建的 torch，其中包括 `torch.distributed.ring_exchange`。
- **类型**: 布尔值。
- **默认值**: `False`

#### `deallocate_pipeline_outputs`
- **描述**: 如果为真，在数据发送到下一个管道阶段后将输出数据释放。有助于节省内存，当不使用管道并行时无效。
- **类型**: 布尔值。
- **默认值**: `False`

#### `defer_embedding_wgrad_compute`
- **描述**: 如果为真，在管道冲刷发生时推迟嵌入层 WGRAD GEMM 的计算，从而使我们能够隐藏管道冲刷延迟。默认为 False。
- **类型**: 布尔值。
- **默认值**: `False`

#### `wgrad_deferral_limit`
- **描述**: 此值调整需要推迟到管道冲刷的嵌入层权重梯度计算的微批次数量，如果 `defer_embedding_wgrad_compute` 为 False，则此参数无效。默认为 0，意味着推迟所有微批次。
- **类型**: 整数。
- **默认值**: `0`

#### `pipeline_model_parallel_split_rank`
- **描述**: 如果为整数，指定在模型中同时包含编码器和解码器（例如，T5）的情况下，编码器和解码器应该在哪个等级分割。如果为 None，则忽略。
- **类型**: 整数（可选）。
- **默认值**: `None`

### 参数组：CPU Offloading

#### `cpu_offloading`
- **描述**: 当设置为 True 时，所有激活都将异步卸载到 CPU。
- **类型**: 布尔值。
- **默认值**: `False`

#### `cpu_offloading_num_layers`
- **描述**: 指定需要将激活卸载的变换器层数量。
- **类型**: 整数。
- **默认值**: `0`

#### `_cpu_offloading_context`
- **描述**: 仅供内部使用，用户不应设置。TODO: 需要在可能时移动到适当的位置。
- **类型**: 上下文管理器。
- **默认值**: `None`

#### `cpu_offloading_activations`
- **描述**: 如果为真，将激活卸载到 CPU。
- **类型**: 布尔值。
- **默认值**: `True`

#### `cpu_offloading_weights`
- **描述**: 如果为真，将权重卸载到 CPU。
- **类型**: 布尔值。
- **默认值**: `True`

### 参数组：Timing

#### `barrier_with_L1_time`
- **描述**: 如果为真，使用具有一级时间测量的屏障。用户需要确保使用他们的计时器调用屏障不会导致挂起。例如，如果用户添加一个不是所有等级都调用的一级计时器，可能会发生这种情况。
- **类型**: 布尔值。
- **默认值**: `True`