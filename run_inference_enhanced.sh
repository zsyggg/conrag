#!/bin/bash

# This script reruns inference (baseline and enhanced) for the ARC_Challenge dataset
# using specified Llama3 models (70B and 8B) via Ollama.
# It calls the Python inference scripts that have been updated to better handle ARC Challenge.

# --- General Configuration ---
BASE_DIR="/workspace/conRAG" # 项目根目录
DATA_BASE_DIR="${BASE_DIR}/data"
RESULTS_BASE_DIR="${BASE_DIR}/results"

DATASET_NAME="arc_challenge"
TASK_TYPE_PROMPT="arc_challenge" # 用于推理脚本中的 --task 参数

# 输入的增强共识文件 (已在阶段一生成)
ENHANCED_CONSENSUS_FILE="${DATA_BASE_DIR}/${DATASET_NAME}/arc_challenge_enhanced_consensus_evidence.jsonl"

# Ollama Configuration
OLLAMA_BASE_URL="http://172.18.147.77:11434"
MAX_NEW_TOKENS_INFERENCE=512 # LLM生成新token的最大数量
NUM_SAMPLES_INFERENCE=-1    # 处理的样本数量 (-1 表示全部)

# --- 定义要测试的Llama模型 ---
# 格式: "ollama_model_name_in_api,tag_for_filename"
declare -a ollama_models_to_run=(
    "llama3:70b,llama3_70b"
    "llama3:8b,llama3_8b"
)

# --- 创建结果目录 ---
mkdir -p "${RESULTS_BASE_DIR}/${DATASET_NAME}"

echo "========================================================================"
echo "Rerunning Inference for Dataset: ${DATASET_NAME} (Task Type: ${TASK_TYPE_PROMPT})"
echo "========================================================================"

# 检查增强共识文件是否存在
if [ ! -f "${ENHANCED_CONSENSUS_FILE}" ]; then
    echo "ERROR: Enhanced consensus file for ${DATASET_NAME} not found at ${ENHANCED_CONSENSUS_FILE}"
    echo "Please ensure it was generated successfully."
    exit 1
fi

# --- 循环遍历模型并运行推理 ---
for model_config in "${ollama_models_to_run[@]}"; do
    IFS=',' read -r OLLAMA_MODEL_NAME OLLAMA_MODEL_TAG <<< "$model_config"

    echo ""
    echo "------------------------------------------------------------------------"
    echo "Using Ollama Model: ${OLLAMA_MODEL_NAME} (Tag: ${OLLAMA_MODEL_TAG}) for ${DATASET_NAME}"
    echo "------------------------------------------------------------------------"

    # --- 运行基线RAG推理 ---
    BASELINE_OUTPUT_JSONL="${RESULTS_BASE_DIR}/${DATASET_NAME}/${DATASET_NAME}_baseline_ollama_${OLLAMA_MODEL_TAG}.jsonl"
    echo "Rerunning Baseline RAG for ${DATASET_NAME} with ${OLLAMA_MODEL_TAG}..."
    # 调用更新后的 inference_baseline_rag.py
    python "${BASE_DIR}/inference_baseline_rag.py" \
      --input_file "${ENHANCED_CONSENSUS_FILE}" \
      --output_file "${BASELINE_OUTPUT_JSONL}" \
      --ollama_base_url "${OLLAMA_BASE_URL}" \
      --ollama_model_name "${OLLAMA_MODEL_NAME}" \
      --task "${TASK_TYPE_PROMPT}" \
      --max_new_tokens ${MAX_NEW_TOKENS_INFERENCE} \
      --num_samples ${NUM_SAMPLES_INFERENCE}
    if [ $? -ne 0 ]; then 
        echo "ERROR during Baseline RAG for ${DATASET_NAME} with ${OLLAMA_MODEL_TAG}"
    else 
        echo "Baseline RAG output: ${BASELINE_OUTPUT_JSONL}"
    fi

    # --- 运行增强RAG推理 ---
    ENHANCED_OUTPUT_JSONL="${RESULTS_BASE_DIR}/${DATASET_NAME}/${DATASET_NAME}_enhanced_ollama_${OLLAMA_MODEL_TAG}.jsonl"
    echo "Rerunning Enhanced RAG for ${DATASET_NAME} with ${OLLAMA_MODEL_TAG}..."
    # 调用更新后的 inference_with_enhanced_rag.py
    python "${BASE_DIR}/inference_with_enhanced_rag.py" \
      --input_file "${ENHANCED_CONSENSUS_FILE}" \
      --output_file "${ENHANCED_OUTPUT_JSONL}" \
      --ollama_base_url "${OLLAMA_BASE_URL}" \
      --ollama_model_name "${OLLAMA_MODEL_NAME}" \
      --task "${TASK_TYPE_PROMPT}" \
      --max_new_tokens ${MAX_NEW_TOKENS_INFERENCE} \
      --num_samples ${NUM_SAMPLES_INFERENCE}
    if [ $? -ne 0 ]; then 
        echo "ERROR during Enhanced RAG for ${DATASET_NAME} with ${OLLAMA_MODEL_TAG}"
    else 
        echo "Enhanced RAG output: ${ENHANCED_OUTPUT_JSONL}"
    fi
    echo "------------------------------------------------------------------------"
done # 结束模型循环

echo "--- Rerun Inference for ARC Challenge with specified models complete! ---"
echo "You can now run the evaluation script for these newly generated results."
