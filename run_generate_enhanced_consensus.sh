#!/bin/bash
# 这个脚本使用优化后的逻辑，为多个数据集 (POPQA, ARC-Challenge, BIO) 生成共识和额外证据。

# --- 配置 ---
BASE_DIR="/workspace/conRAG" # 项目根目录

# --- 模型路径 (所有数据集通用) ---
# 你已经微调好的T5共识模型路径
PRETRAINED_T5_CONSENSUS_MODEL_DIR="/workspace/conRAG/models/t5_consensus_finetuned_popqa_500/best_model_epoch_3_loss_0.5096"
# Sentence Transformer 模型 (本地路径或Hugging Face模型名)
ST_MODEL_NAME="/workspace/all-MiniLM-L6-v2"

# --- 要处理的数据集列表 ---
# 确保你的数据目录中有对应的子目录 (例如 /data/popqa, /data/arc_challenge)
declare -a datasets_to_process=("popqa" "arc_challenge" "bio")

# --- GPU 配置 ---
PROCESSING_GPU_ID=0 # 根据需要调整 (例如 0, 1, 2, 3)
export CUDA_VISIBLE_DEVICES=${PROCESSING_GPU_ID}
echo "全局设置: 使用 GPU ${CUDA_VISIBLE_DEVICES} 进行证据生成。"

# --- 核心参数 ---
# 对与共识相似的句子的惩罚权重。值越高，选出的证据与共识差异越大。
ORTHOGONALITY_PENALTY=0.7

# --- 循环处理每个数据集 ---
for DATASET_NAME in "${datasets_to_process[@]}"; do
    echo ""
    echo "========================================================================"
    echo "--- 开始处理数据集: ${DATASET_NAME} ---"
    echo "========================================================================"

    # --- 动态设置文件路径 (已修复为指向 /data 目录) ---
    # 假设你的检索文件遵循 /data/{dataset_name}/{dataset_name}_retrieved.jsonl 格式
    INPUT_DATA_DIR="${BASE_DIR}/data/${DATASET_NAME}"
    INPUT_RETRIEVED_FILE="${INPUT_DATA_DIR}/${DATASET_NAME}_retrieved.jsonl"
    
    # 定义输出目录和文件
    OUTPUT_ENHANCED_FILE="${INPUT_DATA_DIR}/${DATASET_NAME}_enhanced_consensus_evidence.jsonl"

    # --- Step 1: 检查输入文件 ---
    if [ ! -f "${INPUT_RETRIEVED_FILE}" ]; then
        echo "错误: ${DATASET_NAME} 的输入文件未找到，已跳过: ${INPUT_RETRIEVED_FILE}"
        echo "请确保该文件存在，或者在脚本中调整文件名。"
        continue # 跳过当前数据集，继续下一个
    fi
    echo "找到输入文件: ${INPUT_RETRIEVED_FILE}"

    # --- Step 2: 检查模型路径 ---
    if [ ! -d "${PRETRAINED_T5_CONSENSUS_MODEL_DIR}" ]; then
        echo "错误: 预训练T5共识模型目录未找到: ${PRETRAINED_T5_CONSENSUS_MODEL_DIR}"
        exit 1
    fi
    if [ ! -d "${ST_MODEL_NAME}" ]; then
        echo "错误: SentenceTransformer模型目录未找到: ${ST_MODEL_NAME}"
        exit 1
    fi

    # --- Step 3: 运行优化后的Python脚本生成证据 ---
    echo "正在为 ${DATASET_NAME} 生成增强证据..."
    echo "输出文件将保存至: ${OUTPUT_ENHANCED_FILE}"
    
    python "${BASE_DIR}/generate_enhanced_consensus.py" \
      --input_file "${INPUT_RETRIEVED_FILE}" \
      --output_file "${OUTPUT_ENHANCED_FILE}" \
      --consensus_model_path "${PRETRAINED_T5_CONSENSUS_MODEL_DIR}" \
      --st_model_name "${ST_MODEL_NAME}" \
      --device "cuda" \
      --top_n_orthogonal 2 \
      --orthogonality_penalty ${ORTHOGONALITY_PENALTY}

    if [ $? -ne 0 ] || [ ! -s "${OUTPUT_ENHANCED_FILE}" ]; then
        echo "错误: 为 ${DATASET_NAME} 生成证据失败，或输出文件为空。"
    else
        echo "成功为 ${DATASET_NAME} 生成证据！"
    fi
done

echo ""
echo "--- 所有指定数据集处理完毕 ---"
