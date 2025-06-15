import json
import argparse
import os
import requests
from tqdm import tqdm
import traceback
import re
import sys
sys.path.append('/workspace/conRAG')
from utils import format_arc_choices_for_prompt, postprocess_arc_answer

# 任务指令
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

control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

def format_baseline_prompt(item_index, task, query, consensus_text, original_passages_list, choices_data=None):
    task_specific_instruction = TASK_INST.get(task, "Please answer the following question accurately and comprehensively based on the provided context.")
    
    # ARC Challenge 特殊处理 - 使用CRAG风格
    if task.lower() == "arc_challenge":
        # 1. 准备文档
        documents = ""
        if original_passages_list and any(p.strip() for p in original_passages_list):
            # 只使用前3个文档，避免过长
            valid_passages = [p.strip() for p in original_passages_list if p.strip()][:3]
            documents = ' '.join(valid_passages)
        elif consensus_text and consensus_text.strip() and consensus_text != "Not available.":
            documents = consensus_text
        else:
            documents = "No relevant documents found."
        
        # 2. 格式化选项
        choices_str = format_arc_choices_for_prompt(choices_data) if choices_data else ""
        
        if not choices_str:
            print(f"WARNING: No choices found for item {item_index}")
            # 返回一个基本的prompt
            return f"Question: {query}\nAnswer with A, B, C, or D:"
        
        # 3. CRAG风格的prompt
        prompt = (
            f"Refer to the following documents, follow the instruction and answer the question.\n\n"
            f"Documents: {documents}\n"
            f"Question: {query}\n\n"
            f"Instruction: Given four answer candidates, A, B, C and D, choose the best answer choice.\n"
            f"Choices:{choices_str}"
        )
        
        return prompt
    
    # 其他任务的处理保持原样
    else:
        user_question = query
        original_context_str = "\n\n".join(original_passages_list) if original_passages_list else "Not available."
        consensus_text_str = consensus_text if consensus_text else "Not available."
        
        how_to_answer_instruction_baseline = (
            "Your task is to answer the user's question based on the provided information.\n"
            "Use \"Consensus\" and \"Original Retrieved Documents\" as your primary sources."
        )
        
        prompt_parts = [
            f"{task_specific_instruction}\n\n",
            f"Question: {user_question}\n\n",
            f"### Contextual Information Provided:\n\n",
            f"#### Original Retrieved Documents:\n{original_context_str}\n\n",
            f"#### Consensus:\n{consensus_text_str}\n\n",
            f"### Instructions for Answering:\n{how_to_answer_instruction_baseline}\n\n",
            f"### Answer:\n"
        ]
        
        return "".join(prompt_parts)

_printed_messages_baseline = set()
def print_once_baseline(message):
    if message not in _printed_messages_baseline:
        print(message)
        _printed_messages_baseline.add(message)

def postprocess_answer_baseline(answer, task=""):
    # 清理控制tokens
    for token in control_tokens:
        answer = answer.replace(token, "")
    answer = answer.replace("</s>", "").replace("<|endoftext|>", "").strip()
    
    # ARC特殊处理
    if task.lower() == "arc_challenge":
        return postprocess_arc_answer(answer)
    else:
        # 其他任务的处理
        if "### Answer:" in answer:
            answer = answer.split("### Answer:")[-1].strip()
        elif "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        answer = "\n".join(line.strip() for line in answer.splitlines() if line.strip())
        return answer if answer else ""

def generate_ollama_response_baseline(prompt_text, ollama_base_url, ollama_model_name, max_new_tokens=512, task=""):
    response_content = "ERROR_DURING_OLLAMA_GENERATION_BASELINE"
    api_url = f"{ollama_base_url.rstrip('/')}/api/generate"
    
    generate_options = {
        "num_predict": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.95
    }
    
    # ARC任务的特殊设置
    if task.lower() == "arc_challenge":
        generate_options["num_predict"] = 50  # ARC只需要一个字母答案
        generate_options["temperature"] = 0.0  # 确定性输出
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
        
        response_content = postprocess_answer_baseline(generated_text, task=task)
        return response_content
    except requests.exceptions.RequestException as e:
        print(f"ERROR in generate_ollama_response_baseline: {type(e).__name__} - {e}")
        return f"ERROR_OLLAMA_API_REQUEST_BASELINE: {type(e).__name__}"
    except Exception as e:
        print(f"ERROR in generate_ollama_response_baseline: {type(e).__name__} - {e}")
        return f"ERROR_DURING_OLLAMA_GENERATION_UNSPECIFIED_BASELINE: {type(e).__name__}"

def main():
    parser = argparse.ArgumentParser(description="Generate final answers (JSONL output) for BASELINE RAG using Ollama API.")
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
        print(f"Created output directory: {output_dir}")
    
    processed_records_baseline = []
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            all_lines = infile.readlines()
            lines_to_process = all_lines
            if args.num_samples > 0 and args.num_samples < len(all_lines):
                lines_to_process = all_lines[:args.num_samples]
                print(f"Processing the first {args.num_samples} samples for baseline.")
            else:
                print(f"Processing all {len(all_lines)} samples for baseline.")

            for i, line in enumerate(tqdm(lines_to_process, desc=f"Generating Baseline Responses (Ollama, Task: {args.task})")):
                output_record = {}
                data = None
                generated_answer_for_record = "NOT_GENERATED_YET_BASELINE"
                
                try:
                    data = json.loads(line.strip())
                    
                    # 处理不同的query字段名
                    query = data.get('query', data.get('question', 'QueryFieldMissingInInput'))
                    
                    # 获取choices（仅ARC任务需要）
                    choices_data = data.get('choices', None) if args.task.lower() in ["arc_challenge", "arc_easy"] else None
                    
                    # 获取其他字段
                    original_passages = data.get('original_passages', data.get('passages', []))
                    consensus = data.get('consensus', 'ConsensusMissingInInput')
                    
                    # 构建输出记录
                    output_record = {
                        'query': query,
                        'original_passages': original_passages,
                        'consensus': consensus,
                        'input_line_index': i
                    }
                    
                    if choices_data:
                        output_record['choices'] = choices_data
                    
                    # 检查是否应该跳过
                    skip_keywords = ["ERROR_PROCESSING_ITEM", "NO_PASSAGES_PROVIDED", "NO_VALID_PASSAGES_PROVIDED",
                                   "NO_VALID_PASSAGES_AFTER_FILTERING", "NO_PASSAGES_PROVIDED_TO_T5",
                                   "PASSAGES_EMPTY_AFTER_FILTERING", "CONSENSUS_GENERATION_FAILED"]
                    
                    if consensus in skip_keywords:
                        generated_answer_for_record = f"SKIPPED_DUE_TO_ERROR: {consensus}"
                    else:
                        # 生成prompt
                        prompt = format_baseline_prompt(i, args.task, query, consensus, original_passages, choices_data)
                        
                        # 打印第一个样例
                        if i == 0 or (i == 0 and args.task.lower() == "arc_challenge"):
                            print_once_baseline(f"\nPrompt example (Index: {i}, Task: {args.task}):\n{prompt}\n")
                        
                        # 生成答案
                        generated_answer_for_record = generate_ollama_response_baseline(
                            prompt, args.ollama_base_url, args.ollama_model_name, args.max_new_tokens, task=args.task
                        )
                    
                    output_record['generated_answer'] = generated_answer_for_record
                    
                    # 如果有answerKey，也保存（用于调试）
                    if 'answerKey' in data:
                        output_record['answerKey'] = data['answerKey']
                    
                    processed_records_baseline.append(output_record)

                except json.JSONDecodeError:
                    print(f"JSON decode error at line {i}")
                    processed_records_baseline.append({
                        'input_line_raw': line.strip(),
                        'generated_answer': "SKIPPED_JSON_ERROR_INPUT",
                        'input_line_index': i
                    })
                except Exception as e:
                    print(f"Error processing line {i}: {type(e).__name__} - {e}")
                    error_record = {
                        'query': data.get('query', 'Unknown') if data else 'Unknown',
                        'input_line_index': i,
                        'generated_answer': f"ERROR_IN_PROCESSING: {type(e).__name__}"
                    }
                    processed_records_baseline.append(error_record)
                    
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return
    except Exception as e:
        print(f"Critical error: {type(e).__name__} - {e}")
        traceback.print_exc()
        return

    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for record in processed_records_baseline:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Baseline response generation complete. Output saved to {args.output_file}")
    print(f"Total processed: {len(processed_records_baseline)} records")

if __name__ == '__main__':
    main()