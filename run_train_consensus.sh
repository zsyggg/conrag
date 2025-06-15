#!/bin/bash

# 脚本用于训练T5共识器模型

# --- 请根据您的实际路径和偏好修改以下变量 ---

# 训练数据文件的路径 (由 openai_generate_consensus_data.py 生成)
TRAIN_DATA_FILE="/workspace/conRAG/data/popqa/popqa_consensus_for_t5_training_gpt4o_500_samples.txt"

# 微调后的模型保存路径
SAVE_MODEL_PATH="/workspace/conRAG/models/t5_consensus_finetuned_popqa_500"

# 训练参数 (您可以根据需要调整)
BATCH_SIZE=4
NUM_EPOCHS=3
LEARNING_RATE=3e-5
SEED=42

# --- 指定要使用的单个GPU (例如，使用第一个GPU: 0) ---
export CUDA_VISIBLE_DEVICES=3
# 如果您想使用第二个GPU，可以设置为 export CUDA_VISIBLE_DEVICES=1, 以此类推。

# (可选) 如果之前的NCCL错误仍然出现，可以保留NCCL_DEBUG设置
# export NCCL_DEBUG=INFO

# --- 执行训练命令 ---
echo "Starting T5 consensus model training on a single specified GPU..."
echo "CUDA_VISIBLE_DEVICES is set to: ${CUDA_VISIBLE_DEVICES}"
echo "Training data file: ${TRAIN_DATA_FILE}"
echo "Model save path: ${SAVE_MODEL_PATH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Number of epochs: ${NUM_EPOCHS}"

# 创建保存模型的目录 (如果不存在)
mkdir -p ${SAVE_MODEL_PATH}

python train_consensus.py \
  --train_file "${TRAIN_DATA_FILE}" \
  --save_path "${SAVE_MODEL_PATH}" \
  --batch_size ${BATCH_SIZE} \
  --num_epochs ${NUM_EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --seed ${SEED}

echo "Training finished."
echo "Finetuned model saved to: ${SAVE_MODEL_PATH}"
