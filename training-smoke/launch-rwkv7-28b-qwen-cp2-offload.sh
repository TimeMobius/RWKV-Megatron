#!/usr/bin/env bash
# 8 卡 RWKV7 28B + Qwen tokenizer，TP2 × PP2 × CP2 + SP；50% Adam CPU offload。
set -eo pipefail

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR=/mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test/training-smoke/cache/inductor-py312
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-http://dian.sea-group.org:9098}"
export WANDB_ENTITY="${WANDB_ENTITY:-mobius}"
export HTTPS_PROXY="${RWKV_PROXY_URL:-http://192.168.9.1:7893}"
export HTTP_PROXY="$HTTPS_PROXY"
export https_proxy="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export SOCKS_PROXY="${RWKV_SOCKS_PROXY:-socks5://192.168.9.1:7893}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/mnt/data/anaconda3/envs/train_mobius_py312
export LD_LIBRARY_PATH=/mnt/data/anaconda3/envs/train_mobius_py312/lib:/mnt/data/anaconda3/envs/train_mobius_py312/targets/x86_64-linux/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cublas/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cusparse/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cusolver/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/cudnn/lib:/mnt/data/anaconda3/envs/train_mobius_py312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius_py312
set -u

cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-upstream-test
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p /mnt/data/Logs/outputs

phase="${1:-qwen-rwkv28b-seq4096-cp2-offload-$(date +%Y%m%d-%H%M%S)}"
config="${2:-$PWD/configs/config-rwkv7-28b-qwen3.8-0056-cp2-offload.yaml}"

python - "$config" <<'CHECK_CONFIG'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
c = yaml.safe_load(config_path.read_text())
model_parallel = c.get("model_parallel", {})
if model_parallel.get("context_parallel_size") != 2:
    raise SystemExit("CP2 启动脚本要求 model_parallel.context_parallel_size: 2")
if not model_parallel.get("sequence_parallel"):
    raise SystemExit("此配置要求 sequence_parallel: true")
if not c.get("optimizer_cpu_offload") or not c.get("use_precision_aware_optimizer"):
    raise SystemExit("28B 八卡 CP2 要求启用分布式 precision-aware Adam CPU offload")
if not c.get("use_distributed_optimizer"):
    raise SystemExit("此配置要求 use_distributed_optimizer: true")
if model_parallel.get("tensor_model_parallel_size") != 2 or model_parallel.get("pipeline_model_parallel_size") != 2:
    raise SystemExit("8 卡 CP2 配置要求 TP2 × PP2 × CP2")
if c.get("save") and not c.get("load"):
    out = Path(c["save"])
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"检查点目录已非空：{out}。续训请设置 load；新实验请更改 save。")
for key in ("save", "wandb_save_dir", "tensorboard_dir", "data_cache_path"):
    if c.get(key):
        Path(c[key]).mkdir(parents=True, exist_ok=True)
CHECK_CONFIG

export RWKV_GRAD_AUDIT_INTERVAL="${RWKV_GRAD_AUDIT_INTERVAL:-100}"
export RWKV_GRAD_AUDIT_OUTPUT="/mnt/data/Logs/outputs/${phase}-grad-audit.jsonl"
entry=pretrain_rwkv7.py
if [[ "${RWKV_GRAD_AUDIT:-0}" == "1" ]]; then
  entry=training-smoke/audit/grad-growth-20261013/diagnostic_train.py
fi

torchrun --master_port "${REFERENCE_MASTER_PORT:-51530}" --nproc-per-node 8 --nnodes 1 "$entry" \
  --rwkv-config "$config" 2>&1 | tee "/mnt/data/Logs/outputs/${phase}.log"
