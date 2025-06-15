#!/bin/bash

# This script runs the entire pipeline for POPQA, ARC-Challenge, and BIO datasets.
# It first generates enhanced consensus with additional evidence, then performs
# baseline and enhanced RAG inference using the llama3 8B model via Ollama.

BASE_DIR="/workspace/conRAG"          # adjust if your project root differs
DATA_BASE_DIR="${BASE_DIR}/data"
RESULTS_BASE_DIR="${BASE_DIR}/results"

# Ollama configuration (use your server URL)
OLLAMA_BASE_URL="http://172.18.147.77:11434"
OLLAMA_MODEL="llama3:8b"
MAX_NEW_TOKENS=512
NUM_SAMPLES=-1  # set >0 to limit processed records

datasets=("popqa" "arc_challenge" "bio")

# -----------------------------------------------------------------------------
# Step 1: generate consensus and additional evidence for each dataset
# -----------------------------------------------------------------------------

bash run_generate_enhanced_consensus.sh

# -----------------------------------------------------------------------------
# Step 2: generate final answers using the 8B model for baseline and enhanced RAG
# -----------------------------------------------------------------------------

for dataset in "${datasets[@]}"; do
    echo "\n========================================"
    echo "Processing dataset: ${dataset}"
    echo "========================================"

    INPUT_FILE="${DATA_BASE_DIR}/${dataset}/${dataset}_enhanced_consensus_evidence.jsonl"
    BASELINE_OUTPUT="${RESULTS_BASE_DIR}/${dataset}/${dataset}_baseline_ollama_llama3_8b.jsonl"
    ENHANCED_OUTPUT="${RESULTS_BASE_DIR}/${dataset}/${dataset}_enhanced_ollama_llama3_8b.jsonl"

    mkdir -p "${RESULTS_BASE_DIR}/${dataset}"

    # Baseline RAG generation
    python inference_baseline_rag.py \
      --input_file "${INPUT_FILE}" \
      --output_file "${BASELINE_OUTPUT}" \
      --ollama_base_url "${OLLAMA_BASE_URL}" \
      --ollama_model_name "${OLLAMA_MODEL}" \
      --task "${dataset}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --num_samples ${NUM_SAMPLES}

    # Enhanced RAG generation
    python inference_with_enhanced_rag.py \
      --input_file "${INPUT_FILE}" \
      --output_file "${ENHANCED_OUTPUT}" \
      --ollama_base_url "${OLLAMA_BASE_URL}" \
      --ollama_model_name "${OLLAMA_MODEL}" \
      --task "${dataset}" \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --num_samples ${NUM_SAMPLES}

done

echo "\nPipeline complete. Results saved to ${RESULTS_BASE_DIR}."

