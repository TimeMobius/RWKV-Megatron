#!/bin/bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=24
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export DEBUG_RANK=0

export SOCKS_PROXY="socks5://192.168.9.1:7893"
export HTTPS_PROXY="http://192.168.9.1:7893"
export HTTP_PROXY="http://192.168.9.1:7893"
# export CUDA_VISIBLE_DEVICES=4,5,6,7



source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate train_mobius



# # 设置conda的CUDA路径
# export CONDA_CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux

# # 配置环境变量
# export CUDA_HOME=$CONDA_PREFIX
# export C_INCLUDE_PATH=$CONDA_CUDA_PATH/include:$C_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=$CONDA_CUDA_PATH/include:$CPLUS_INCLUDE_PATH
# export LD_LIBRARY_PATH=$CONDA_CUDA_PATH/lib:$LD_LIBRARY_PATH
# export PATH=$CUDA_HOME/bin:$PATH




# # 创建临时目录存放软链接
# TMP_DIR=$(mktemp -d)

# # 在临时目录中创建gcc和g++的软链接
# ln -s $(which gcc-13) "${TMP_DIR}/gcc"
# ln -s $(which g++-13) "${TMP_DIR}/g++"

# # 将临时目录添加到PATH的最前面
# export PATH="${TMP_DIR}:${PATH}"

# # 验证当前gcc版本
# echo "当前使用的GCC版本："
# gcc --version








cd /mnt/data/Codes/RWKV/RWKV-Megatron-H200/RWKV/RWKV-Megatron-newest
torchrun  --master_port 51281 --nproc-per-node 8 --nnodes 1 pretrain_rwkv7.py --yaml-cfg /mnt/data/Configs/config-mobius-r7-pretrain-14b-megatron.yaml | tee /mnt/data/Logs/outputs/output_wandb_test.log