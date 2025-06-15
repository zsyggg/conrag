import json
import argparse
import os
import requests
from tqdm import tqdm
import traceback
import re
import sys

# 将项目根目录添加到Python路径，以便导入utils
# 请确保这个路径是正确的
sys.path.append('/workspace/conRAG')
from utils import format_arc_choices_for_prompt, postprocess_arc_answer

# 任务指令字典
TASK_INST = {
    "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable, and engaging response.",
    "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
    "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
    "obqa": "Given four answer candidates, A, B, C, and D, choose the best answer choice.",
    "arc_easy": "Given four answer candidates, A, B, C, and D, choose the best answer choice.",
    "arc_challenge": "Given four answer candidates, A, B, C, and D, choose the best answer choice.",
    "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
    "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
    "popqa": "Answer the following question based on the provided information.",
    "bio": "Generate a comprehensive biography based on the provided information."
}

# 控制Token列表，用于后处理清理
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

# Prompts in this script intentionally exclude raw retrieval passages and any
# additional evidence to minimise noise.

def format_enhanced_prompt(item_index, task, query, consensus_text, additional_evidence_list, original_passages_list, choices_data=None):
    """Create the prompt for the enhanced RAG method.

    Retrieval passages and additional evidence are omitted from the prompt to
    reduce noise. Only the consensus summary is used as supporting context.  The
    parameters ``additional_evidence_list`` and ``original_passages_list`` are
    kept for API compatibility but ignored here.
    """
    system_intro = "You are a helpful and precise assistant. Your task is to answer the following question based on the provided context. Be critical and verify information."

    # --- 针对 ARC Challenge 的特殊、严格的Prompt ---
    if task.lower() == "arc_challenge":
        # Only include the consensus text. Additional evidence is ignored.
        context_parts = []
        if consensus_text and consensus_text.strip() and consensus_text != "Not available.":
            context_parts.append(f"- [Consensus]: {consensus_text}")
        context_str = "\n".join(context_parts) if context_parts else "No relevant context found."
        choices_str = format_arc_choices_for_prompt(choices_data) if choices_data else ""
        if not choices_str:
            return f"Question: {query}\nAnswer with only the letter A, B, C, or D."
        prompt = f"""{system_intro}

### INSTRUCTIONS ###
1.  Carefully read the question and the multiple-choice options.
2.  Analyze the provided context ([Consensus] and [Additional Evidence]) to find the most accurate answer.
3.  **Crucially, be aware that the context might contain misleading information or distractors.** Do not blindly trust it.
4.  Your final answer must be a single capital letter (A, B, C, or D) corresponding to the best choice. Do not provide any explanation.

### QUESTION ###
{query}

### CONTEXT ###
{context_str}

### CHOICES ###
{choices_str}

### FINAL ANSWER (A, B, C, or D) ###
"""
        return prompt
    # --- 针对其他任务的通用增强Prompt ---
    else:
        consensus_str = consensus_text if consensus_text else "Not available."
        prompt_parts = [
            f"{system_intro}\n\n",
            f"### Question:\n{query}\n\n",
            f"### Context:\n",
            f"#### Consensus:\n{consensus_str}\n\n",
            f"### Instruction:\n"
            f"Based on the provided consensus, provide a direct and comprehensive answer to the question.\n\n",
            f"### Answer:\n"
        ]
        return "".join(prompt_parts)


_printed_messages = set()
def print_once(message):
    if message not in _printed_messages:
        print(message)
        _printed_messages.add(message)

def postprocess_answer(answer, task=""):
    """后处理模型生成的答案"""
    for token in control_tokens:
        answer = answer.replace(token, "")
    answer = answer.replace("</s>", "").replace("<|endoftext|>", "").strip()
    
    if task.lower() == "arc_challenge":
        return postprocess_arc_answer(answer)
    else:
        if "### Answer:" in answer:
            answer = answer.split("### Answer:")[-1].strip()
        elif "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        answer = "\n".join(line.strip() for line in answer.splitlines() if line.strip())
        return answer if answer else ""

def generate_ollama_response(prompt_text, ollama_base_url, ollama_model_name, max_new_tokens=512, task=""):
    """通过Ollama API调用LLM进行推理"""
    api_url = f"{ollama_base_url.rstrip('/')}/api/generate"
    
    generate_options = {
        "num_predict": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.95
    }
    
    if task.lower() == "arc_challenge":
        generate_options["num_predict"] = 50
        generate_options["temperature"] = 0.0 # 确定性输出
        generate_options["top_p"] = 1.0

    payload = {
        "model": ollama_model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": generate_options
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data.get("response", "")
        return postprocess_answer(generated_text, task=task)
    except requests.exceptions.RequestException as e:
        print(f"ERROR in generate_ollama_response: {type(e).__name__} - {e}")
        return f"ERROR_OLLAMA_API_REQUEST: {type(e).__name__}"
    except Exception as e:
        print(f"ERROR in generate_ollama_response: {type(e).__name__} - {e}")
        return f"ERROR_DURING_OLLAMA_GENERATION_UNSPECIFIED: {type(e).__name__}"

def main():
    parser = argparse.ArgumentParser(description="Generate final answers using LLM with enhanced consensus and evidence via Ollama API.")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--ollama_base_url', type=str, required=True)
    parser.add_argument('--ollama_model_name', type=str, required=True)
    parser.add_argument('--task', type=str, default="popqa")
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=-1)
    args = parser.parse_args()

    print(f"Using Ollama API at: {args.ollama_base_url} with model: {args.ollama_model_name} for task: {args.task}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_records = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            all_lines = infile.readlines()
            lines_to_process = all_lines if args.num_samples <= 0 else all_lines[:args.num_samples]
            print(f"Processing {len(lines_to_process)} samples.")

            for i, line in enumerate(tqdm(lines_to_process, desc=f"Generating Enhanced Responses (Task: {args.task})")):
                try:
                    data = json.loads(line.strip())
                    query = data.get('query', data.get('question', 'QueryFieldMissingInInput'))
                    choices_data = data.get('choices') if args.task.lower().startswith("arc") else None
                    original_passages = data.get('original_passages', data.get('passages', []))
                    consensus = data.get('consensus', 'ConsensusMissingInInput')
                    additional_evidence = data.get('additional_evidence', [])
                    
                    output_record = data.copy() # 复制原始数据所有字段
                    output_record['input_line_index'] = i

                    skip_keywords = ["ERROR_PROCESSING_ITEM", "NO_VALID_PASSAGES_PROVIDED", "CONSENSUS_GENERATION_FAILED"]
                    if any(keyword in consensus for keyword in skip_keywords):
                        generated_answer = f"SKIPPED_DUE_TO_ERROR: {consensus}"
                    else:
                        prompt = format_enhanced_prompt(i, args.task, query, consensus, additional_evidence, 
                                                      original_passages, choices_data)
                        if i == 0:
                            print_once(f"\nEnhanced prompt example (Index: {i}, Task: {args.task}):\n{prompt}\n")
                        
                        generated_answer = generate_ollama_response(
                            prompt, args.ollama_base_url, args.ollama_model_name, args.max_new_tokens, task=args.task
                        )
                    
                    output_record['generated_answer'] = generated_answer
                    processed_records.append(output_record)

                except Exception as e:
                    print(f"Error processing line {i}: {type(e).__name__} - {e}")
                    traceback.print_exc()
                    processed_records.append({'input_line_raw': line.strip(), 'generated_answer': f"ERROR_IN_PROCESSING: {type(e).__name__}", 'input_line_index': i})
    
    except Exception as e:
        print(f"Critical error: {type(e).__name__} - {e}")
        traceback.print_exc()

    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for record in processed_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\nEnhanced response generation complete. Output saved to {args.output_file}")
    print(f"Total processed: {len(processed_records)} records")

if __name__ == '__main__':
    main()
