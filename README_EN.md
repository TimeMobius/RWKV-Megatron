# RWKV-Megatron

<div align="center">

<img src="/assets/logo.png" width="300px">

**High-Performance RWKV Large Model Training Framework Built on Megatron-Core**

[![license](https://img.shields.io/github/license/BlinkDL/RWKV-LM)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

English | [简体中文](README.md)
</div>

## 📖 Project Overview

RWKV-Megatron is a distributed training framework for RWKV models built on NVIDIA Megatron-Core. It integrates RWKV's linear-time modeling path with Megatron-Core's training infrastructure, and provides a unified stack for model implementation, configuration management, checkpoint handling, weight initialization, and format conversion.

The objective of this repository is not merely to make RWKV runnable, but to provide a practical training system that can be used for medium- to large-scale experiments and engineering workloads, while leveraging Megatron-Core's mature support for parallel training, mixed precision, memory optimization, and distributed optimizers.

**Advantages of RWKV**:
- Combines RNN-like efficient inference with Transformer-like parallel training
- Linear-time modeling path suitable for long-sequence training and deployment scenarios
- Strong engineering trade-off across long-context modeling, memory efficiency, and inference throughput

**Advantages of Megatron-Core**:
- Modular large-model training infrastructure and distributed abstractions
- Mature tensor, pipeline, sequence, and data parallel training support
- Built-in mixed precision, activation recomputation, distributed optimizers, and checkpoint mechanisms

## 🚀 Core Features

### Model Architecture
- **Native RWKV6 / RWKV7 Support**: Dedicated training entry points and model implementations for RWKV6 and RWKV7
- **Megatron-Core-Oriented Integration**: RWKV modules are adapted to Megatron-Core training workflows
- **Modular YAML Configuration**: Model structure, training hyperparameters, parallel settings, and logging behaviors are organized through YAML
- **Multiple Initialization Sources**: RWKV7 can be initialized from both `rwkv-lm` and `rwkv-huggingface` weights

### Distributed Training
- **Multi-Dimensional Parallel Training**:
  - **Tensor Parallelism**
  - **Pipeline Parallelism**
  - **Sequence Parallelism**
  - **Data Parallelism**
- **YAML-Driven Training Organization**: Parallel strategies can be adjusted efficiently for different hardware scales

### Performance Optimization
- **Mixed Precision Training**: FP16 and BF16 are supported
- **Activation Recomputation**: Reduces memory footprint during training
- **Distributed Optimizer Support**: Reuses Megatron distributed optimizer paths
- **Muon Optimizer Support**: Muon has been integrated into the underlying optimizer stack for optimizer comparison and experimentation
- **Communication and Computation Optimization Options**: Includes gradient accumulation fusion and communication overlap controls from Megatron-Core

### Interoperability
- **Checkpoint and Weight Conversion Toolchain**:
  - Convert Megatron distributed checkpoints into native RWKV weight format

## 💡 Use Cases

- **Large-Scale Pretraining**: Organize RWKV pretraining jobs on multi-GPU or multi-node systems
- **Architecture and Training Comparisons**: Compare different parallel strategies, precision modes, and optimizers
- **Domain Adaptation and Continued Pretraining**: Continue training from existing RWKV checkpoints
- **Weight Conversion and Engineering Integration**: Prepare weights for downstream inference stacks or HuggingFace-based deployment

## 🔧 Quick Start

### Installation and Environment Setup

It is recommended to start from the Conda environment file maintained in this repository:

```bash
# Clone repository and submodules
git clone --recursive https://github.com/shiroko98/RWKV_Megatron.git
cd RWKV_Megatron

# Create and activate environment
conda env create -f environment.yml
conda activate train_rwkv_nvidia_megatron_rwkv7
```

The repository root currently links `megatron` and `fla` to:

```bash
megatron -> 3rdparty/Megatron-LM-0.11.0/megatron
fla      -> 3rdparty/flash-linear-attention-260115/fla
```

If you use a custom environment, make sure the following components are available:
- Python 3.10+
- PyTorch 2.2+
- CUDA / NCCL
- FlashAttention
- DeepSpeed
- Transformers

### Verify the Environment

Use the following script to verify your PyTorch and CUDA setup:

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

## 📘 Usage

Typical workflows include:
1. Data preprocessing
2. Pretraining or continued pretraining
3. Weight export and format conversion

### 1. Data Preprocessing

Training data is expected in loose `jsonl` format, with at least one text field per line:

```json
{"text": "The quick brown fox"}
{"text": "jumps over the lazy dog"}
```

Use `preprocess_data.py` to convert raw text data into Megatron-compatible `bin / idx` files:

```bash
python preprocess_data.py \
  --input /path/to/data.jsonl \
  --output-prefix /path/to/output/train \
  --json-keys text \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /path/to/tokenizer \
  --append-eod
```

The output usually contains:
- `train_text_document.bin`
- `train_text_document.idx`

In training configuration files, `data_path` should point to this prefix without the extension.

### 2. Training RWKV Models

#### YAML Configuration Examples

Several YAML configurations are already provided in [configs](./configs):

- `configs/config-rwkv7-task24-4d-smoke.yaml`
- `configs/config-rwkv7-task23-scratch-init-verify.yaml`
- `configs/config-mobius-r7-pretrain-14b-megatron.yaml`
- `example_config.yaml`

Typical configuration fields include:

```yaml
language_model:
  num_layers: 32
  hidden_size: 8192
  num_attention_heads: 32
  ffn_hidden_size: 28672

model_parallel:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 1
  sequence_parallel: True
  bf16: True

optimizer: adam
lr: 2.5e-4
global_batch_size: 64
seq_length: 4096
```

To enable Muon, switch the optimizer field to:

```yaml
optimizer: muon
```

Muon-related hyperparameters follow Megatron naming conventions, such as `muon_matched_adamw_rms`, `muon_momentum`, and `muon_ns_steps`.

#### Distributed Training Configuration

Different parallel strategies can be organized according to the available hardware:

```python
# 8-GPU Tensor Parallelism
--tensor-model-parallel-size 8 --pipeline-model-parallel-size 1

# 2-GPU Tensor Parallelism × 4-GPU Pipeline Parallelism
--tensor-model-parallel-size 2 --pipeline-model-parallel-size 4

# Enable Sequence Parallelism
--sequence-parallel --tensor-model-parallel-size 8
```

#### Launch Scripts

**RWKV6**

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

**RWKV7**

```bash
#!/bin/bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=24
export CUDA_DEVICE_MAX_CONNECTIONS=1

conda activate train_rwkv_nvidia_megatron_rwkv7
cd /path/to/RWKV_Megatron

torchrun --master_port 51281 --nproc-per-node 8 --nnodes 1 \
  pretrain_rwkv7.py --yaml-cfg /path/to/config_rwkv7.yaml | tee output.log
```

A simple example script is also available:

- `example_pretain_rwkv6.sh`

### 3. Model Weight Conversion

#### 1. Convert Megatron Distributed Checkpoints to Native RWKV Weights

```bash
python rwkv_conversion.py \
  --load-dir /path/to/model \
  --save-dir /path/to/save
```

#### 2. `pth -> HuggingFace` Style Conversion

For `pth -> HuggingFace` style conversion, please refer to the standalone repository:

- [shiroko98/RWKV-7-World-HF-Converter](https://github.com/shiroko98/RWKV-7-World-HF-Converter)

## 🤝 Contributing

Contributions of all kinds are welcome, including but not limited to:

1. Reporting issues and requesting features
2. Improving model implementations, training configurations, or utility scripts
3. Refining documentation and usage guides
4. Sharing training experience, optimization strategies, and experimental observations

## 📚 Citation and Acknowledgments

If you find this repository useful for research or engineering work, please consider citing:

```bibtex
@software{Liu2025RWKV_Megatron,
  title  = {RWKV_Megatron: A high-performance RWKV model training framework based on NVIDIA Megatron-Core},
  author = {Liu, Mingda and Zhang, Guozhen},
  url    = {https://github.com/TimeMobius/RWKV-Megatron},
  month  = May,
  year   = {2025}
}
```

This project builds upon the following open-source projects:

- [Megatron-LM / Megatron-Core](https://github.com/NVIDIA/Megatron-LM)
- [RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

## 📄 License

This project is released under the [LICENSE](LICENSE).
