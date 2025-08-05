#!/bin/bash

set -x                               # enable debug mode to print commands and their arguments as they are executed
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # replace it with your local GPU IDs
export PYTHONUNBUFFERED=1
# export NCCL_P2P_DISABLE=1            # disable NCCL P2P to avoid potential issues with multi-GPU training

# 设置 Hugging Face 镜像和缓存目录
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="./hf_home"
export TRANSFORMERS_CACHE="./hf_home/transformers"
export HF_DATASETS_CACHE="./hf_home/datasets"

# 可选：创建缓存目录（确保目录存在）
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"

# 如果可以正常联网，就不需要替换为本地地址，直接自动huggingface下载即可
MODEL_PATH=XiaomiMiMo/MiMo-VL-7B-RL  # replace it with your local file path

# 修改数据的比例
SPLIT_RATIO=0.5

# rollout的数量
ROLLOUT_N=10

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./data/geometry3k/data@train \
    data.val_files=./data/geometry3k/data@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=XiaomiMiMo_7B_RL_geo_grpo \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=8 \
    worker.rollout.split_ratio=${SPLIT_RATIO} \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.image_text_mixture=False \
    worker.rollout.tensor_parallel_size=8
