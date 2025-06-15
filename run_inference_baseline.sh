#!/bin/bash

# --- Configuration ---
# Input file: This is the SAME input file used by the enhanced RAG,
# as it contains the necessary query, original_passages, and consensus.
# The baseline script will simply ignore the 'additional_evidence' field.
INPUT_ENHANCED_JSONL_FILE="/workspace/conRAG/data/popqa/popqa_enhanced_consensus_evidence.jsonl"

# Output file for final LLM responses for the BASELINE method (JSONL)
OUTPUT_RESPONSES_BASELINE_JSONL_FILE="/workspace/conRAG/results/popqa_final_responses_baseline_hf.jsonl"

# Path to your large language model (self-RAG 7B)
LLM_MODEL_PATH="/workspace/conRAG/self-rag"

TASK_TYPE="popqa"
DEVICE="cuda:2"
MAX_NEW_TOKENS=512
MODEL_PRECISION="fp16"

# Number of samples to process (-1 for all)
# Should match the number you used for the enhanced RAG if you want a direct comparison on the same subset
NUM_SAMPLES_TO_PROCESS=200 # Example: process the first 200 samples

# --- Run the Python script for BASELINE RAG ---
echo "Starting BASELINE RAG response generation (Hugging Face Transformers, JSONL output)..."
echo "Input file: $INPUT_ENHANCED_JSONL_FILE"
echo "Output file: $OUTPUT_RESPONSES_BASELINE_JSONL_FILE"
echo "LLM model: $LLM_MODEL_PATH"
echo "Number of samples to process: $NUM_SAMPLES_TO_PROCESS"

# export CUDA_VISIBLE_DEVICES=0 # Uncomment if needed

python inference_baseline_rag.py \
  --input_file "$INPUT_ENHANCED_JSONL_FILE" \
  --output_file "$OUTPUT_RESPONSES_BASELINE_JSONL_FILE" \
  --llm_model_path "$LLM_MODEL_PATH" \
  --device "$DEVICE" \
  --task "$TASK_TYPE" \
  --max_new_tokens $MAX_NEW_TOKENS \
  --model_precision "$MODEL_PRECISION" \
  --num_samples $NUM_SAMPLES_TO_PROCESS

echo "Baseline RAG response generation finished."
echo "Output saved to: $OUTPUT_RESPONSES_BASELINE_JSONL_FILE"

