import argparse
import json
import os
from tqdm import tqdm
import re
import time # Added for potential delays
from openai import OpenAI # Ensured OpenAI is imported

# --- 从 metrics.py 导入必要的函数 ---
try:
    from metrics import normalize_answer
except ImportError:
    print("Warning: metrics.py not found or normalize_answer not importable. Using basic normalize_answer defined in eval.py.")
    def normalize_answer(s):
        import string # Local import
        def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text): return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

# --- 数据加载函数 ---
def load_jsonl_data(file_path, num_samples=-1):
    records = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return records
    
    print(f"Loading data from JSONL file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples > 0 and i >= num_samples:
                print(f"Loaded the first {num_samples} samples as requested.")
                break
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {i+1} in {file_path}: {line.strip()}")
    print(f"Successfully loaded {len(records)} records from {file_path}.")
    return records

# --- POPQA 评估逻辑 ---
def popqa_loose_match(prediction_str, ground_truth_list_str):
    if not prediction_str or not ground_truth_list_str:
        return False
    norm_prediction = normalize_answer(prediction_str)
    if not norm_prediction:
        return False
    for gt_raw in ground_truth_list_str:
        norm_gt = normalize_answer(str(gt_raw))
        if not norm_gt:
            continue
        if norm_gt in norm_prediction:
            return True
    return False

def evaluate_popqa_jsonl(predictions_jsonl, golden_jsonl):
    loose_match_count = 0
    evaluated_count = 0
    
    min_len = min(len(predictions_jsonl), len(golden_jsonl))
    if len(predictions_jsonl) != len(golden_jsonl):
        print(f"Warning: Aligning predictions ({len(predictions_jsonl)}) and golden ({len(golden_jsonl)}) to {min_len} records.")
    
    predictions_jsonl = predictions_jsonl[:min_len]
    golden_jsonl = golden_jsonl[:min_len]

    if min_len == 0:
        print("No data to evaluate after alignment.")
        return {"accuracy_loose_match": 0.0, "evaluated_count": 0, "total_predictions": 0}

    print(f"Evaluating {len(predictions_jsonl)} prediction/golden pairs for POPQA using Loose Match...")
    for pred_item, gold_item in tqdm(zip(predictions_jsonl, golden_jsonl), total=len(predictions_jsonl), desc="Evaluating POPQA (Loose Match)"):
        generated_answer = pred_item.get("generated_answer", "")
        golden_answers = gold_item.get("answers", [])
        if not isinstance(golden_answers, list):
            golden_answers = [str(golden_answers)] if golden_answers else []
        golden_answers = [str(ga) for ga in golden_answers if isinstance(ga, (str, int, float))]
        if not golden_answers:
            continue
        evaluated_count += 1
        if popqa_loose_match(generated_answer, golden_answers):
            loose_match_count += 1
    accuracy = loose_match_count / evaluated_count if evaluated_count > 0 else 0.0
    results = {"accuracy_loose_match": accuracy, "evaluated_count": evaluated_count, "total_predictions": len(predictions_jsonl)}
    return results

# --- ARC Challenge 评估逻辑 ---
def evaluate_arc_challenge_jsonl(predictions_jsonl, golden_jsonl):
    correct_count = 0
    evaluated_count = 0

    min_len = min(len(predictions_jsonl), len(golden_jsonl))
    if len(predictions_jsonl) != len(golden_jsonl):
        print(f"Warning: Aligning ARC predictions ({len(predictions_jsonl)}) and golden ({len(golden_jsonl)}) to {min_len} records.")

    predictions_jsonl = predictions_jsonl[:min_len]
    golden_jsonl = golden_jsonl[:min_len]
    
    if min_len == 0:
        print("No ARC data to evaluate after alignment.")
        return {"accuracy": 0.0, "evaluated_count": 0, "total_predictions": 0}

    print(f"Evaluating {len(predictions_jsonl)} prediction/golden pairs for ARC Challenge...")
    for pred_item, gold_item in tqdm(zip(predictions_jsonl, golden_jsonl), total=len(predictions_jsonl), desc="Evaluating ARC Challenge"):
        generated_answer_raw = pred_item.get("generated_answer", "")
        generated_answer_match = re.search(r'[A-Z]', generated_answer_raw)
        generated_answer = generated_answer_match.group(0) if generated_answer_match else generated_answer_raw.strip()[:1].upper()

        golden_answer_key = gold_item.get("answerKey", gold_item.get("answer", ""))
        if not golden_answer_key:
            golden_answers_list = gold_item.get("answers", [])
            if isinstance(golden_answers_list, list) and len(golden_answers_list) == 1 and isinstance(golden_answers_list[0], str):
                golden_answer_key = golden_answers_list[0]
            elif isinstance(golden_answers_list, str):
                 golden_answer_key = golden_answers_list

        if not golden_answer_key or not isinstance(golden_answer_key, str):
            continue
        
        evaluated_count += 1
        if normalize_answer(generated_answer) == normalize_answer(golden_answer_key.strip()):
            correct_count += 1

    accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0.0
    results = {"accuracy": accuracy, "evaluated_count": evaluated_count, "total_predictions": len(predictions_jsonl)}
    return results


# --- FActScore 评估逻辑 (BIO Dataset) ---
def initialize_openai_client(api_key, base_url=None, timeout=60):
    if not api_key:
        raise ValueError("OpenAI API key is required for FactScore evaluation.")
    try:
        if base_url:
            print(f"Initializing OpenAI client with custom base_url: {base_url}")
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            print("Initializing OpenAI client with default base_url.")
            client = OpenAI(api_key=api_key, timeout=timeout)
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        raise

def generate_atomic_facts(text, client, model="gpt-3.5-turbo", max_retries=3, retry_delay=5):
    prompt = f"Please identify all atomic factual claims from the following text. An atomic factual claim is a single, standalone piece of information that can be verified as true or false.\n\nText: {text}\n\nExtract each atomic fact as a separate short sentence. Number each fact. If there are no clear factual claims, respond with 'NO_FACTS_FOUND'."
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000
            )
            fact_text = response.choices[0].message.content.strip()
            if "NO_FACTS_FOUND" in fact_text:
                return []
            facts = [re.sub(r'^\s*\d+[.)]?\s*', '', line.strip()) for line in fact_text.split('\n') if line.strip() and re.match(r'^\s*\d+[.)]', line.strip())]
            if facts: return facts
            if fact_text and not facts:
                potential_facts = [f.strip() for f in fact_text.split('\n') if f.strip() and len(f.strip()) > 10]
                if potential_facts: return potential_facts
            if attempt < max_retries - 1: time.sleep(retry_delay)
        except Exception as e:
            print(f"Error generating atomic facts (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1: time.sleep(retry_delay)
    return []


def verify_atomic_fact(fact, reference, client, model="gpt-3.5-turbo", max_retries=3, retry_delay=5):
    prompt = f"Your task is to carefully check if a reference text supports a given factual claim.\n\nFactual claim: \"{fact}\"\nReference text: \"{reference}\"\n\nAnalyze whether the claim is:\n- S (Supported by the reference text)\n- C (Contradicted by the reference text)\n- U (Uncertain - not clearly supported or contradicted by the reference text)\n\nReply with ONLY one letter: S, C, or U."
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            if result and result[0] in ['S', 'C', 'U']: return result[0]
            print(f"Warning: Unexpected verification result '{result}' for fact '{fact}'. Defaulting to 'U'.")
            return 'U'
        except Exception as e:
            print(f"Error verifying atomic fact (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1: time.sleep(retry_delay)
    return 'U'

def calculate_factscore(generated_answers_list, golden_references_list, client, gamma=10, n_samples=None, delay_between_calls=1, verbose=True):
    if not client:
        print("OpenAI client not initialized. Cannot calculate FactScore.")
        return 0.0, 0, 0, 0

    if n_samples and 0 < n_samples < len(generated_answers_list):
        generated_answers_list = generated_answers_list[:n_samples]
        golden_references_list = golden_references_list[:n_samples]
    
    scores, total_facts_overall, supported_facts_overall, num_responses_with_facts = [], 0, 0, 0

    for i, (gen_ans, gold_ref) in enumerate(tqdm(zip(generated_answers_list, golden_references_list), total=len(generated_answers_list), desc="Calculating FactScore")):
        if not gen_ans or not gen_ans.strip():
            scores.append(0.0)
            if verbose: print(f"Sample {i}: Empty generated answer, score 0.")
            continue
        if not gold_ref or not gold_ref.strip():
            scores.append(0.0)
            if verbose: print(f"Sample {i}: Empty golden reference, score 0.")
            continue

        facts = generate_atomic_facts(gen_ans, client)
        if verbose: print(f"Sample {i}: Generated Answer: '{gen_ans[:100]}...', Golden Reference: '{gold_ref[:100]}...', Found {len(facts)} facts.")

        if not facts:
            scores.append(0.0)
            if verbose: print(f"Sample {i}: No facts extracted, score 0.")
            continue
        
        num_responses_with_facts += 1
        total_facts_overall += len(facts)
        supported_count_current_sample = 0
        
        for j, fact in enumerate(facts):
            if verbose: print(f"  Fact {j+1}: '{fact}'")
            verification_result = verify_atomic_fact(fact, gold_ref, client)
            if verbose: print(f"    Verification: {verification_result}")
            if verification_result == 'S':
                supported_count_current_sample += 1
            time.sleep(delay_between_calls)
            
        supported_facts_overall += supported_count_current_sample
        precision = supported_count_current_sample / len(facts) if facts else 0.0
        scores.append(precision)

    if not scores: return 0.0, 0, 0, 0
    
    avg_factscore = sum(scores) / len(scores) if scores else 0.0
    return avg_factscore, total_facts_overall, supported_facts_overall, num_responses_with_facts


def evaluate_bio_jsonl(predictions_jsonl, golden_jsonl, client, args):
    """Evaluates BIO dataset using FActScore."""
    print("Evaluating BIO dataset with FActScore...")
    generated_bios = [item.get('generated_answer', '') for item in predictions_jsonl]
    golden_references = []
    for item in golden_jsonl:
        # --- THIS IS THE CORRECTED PART ---
        # It now checks for 'output' first, which matches your golden file format.
        ref = item.get('output', 
                       item.get('ground_truth_biography', 
                                item.get('answer', 
                                         item.get('text', 
                                                  item.get('reference_bio', '')))))
        golden_references.append(ref)

    if not all(isinstance(bio, str) for bio in generated_bios) or \
       not all(isinstance(ref, str) for ref in golden_references):
        print("Error: BIO evaluation expects a list of strings for generated and golden data.")
        return {"factscore": 0.0}

    if len(generated_bios) != len(golden_references):
        print(f"Warning: Mismatch in number of generated bios ({len(generated_bios)}) and golden references ({len(golden_references)}). Aligning to minimum.")
        min_len = min(len(generated_bios), len(golden_references))
        generated_bios = generated_bios[:min_len]
        golden_references = golden_references[:min_len]
        if min_len == 0:
             return {"factscore": 0.0, "total_facts": 0, "supported_facts": 0, "responses_with_facts": 0}

    factscore, total_facts, supported_facts, responses_with_facts = calculate_factscore(
        generated_bios, 
        golden_references, 
        client, 
        gamma=args.gamma,
        n_samples=len(generated_bios),
        delay_between_calls=args.delay, 
        verbose=args.verbose
    )
    results = {
        "factscore": factscore,
        "total_facts_extracted": total_facts,
        "supported_facts": supported_facts,
        "responses_with_facts": responses_with_facts
    }
    return results

# --- Main Evaluation Logic ---
def evaluate_file_jsonl(golden_filepath, answer_filepath, dataset_type, num_samples=-1, 
                        openai_api_key=None, openai_base_url=None, openai_timeout=60,
                        gamma=10, delay_between_calls=1, verbose=False):
    print(f"--- Evaluating {dataset_type} dataset ---")
    golden_data_list = load_jsonl_data(golden_filepath, num_samples)
    if not golden_data_list:
        print(f"Error: Failed to load golden data from {golden_filepath}. Exiting.")
        return None
    answer_data_list = load_jsonl_data(answer_filepath, num_samples if num_samples > 0 else -1)
    if not answer_data_list:
        print(f"Error: Failed to load answers from {answer_filepath}. Exiting.")
        return None
    
    min_len = min(len(golden_data_list), len(answer_data_list))
    if num_samples > 0:
        min_len = min(min_len, num_samples)

    if len(golden_data_list) > min_len or len(answer_data_list) > min_len:
         print(f"Aligning loaded data to {min_len} samples for evaluation.")
    golden_data_list = golden_data_list[:min_len]
    answer_data_list = answer_data_list[:min_len]

    if not golden_data_list or not answer_data_list:
        print("No data to evaluate after loading and alignment. Exiting.")
        return None

    if dataset_type == 'POPQA':
        return evaluate_popqa_jsonl(answer_data_list, golden_data_list)
    elif dataset_type == 'ARC_Challenge':
        return evaluate_arc_challenge_jsonl(answer_data_list, golden_data_list)
    elif dataset_type == 'BIO':
        if not openai_api_key:
            print("Error: OpenAI API key is required for BIO (FActScore) evaluation.")
            return {"factscore": 0.0, "error": "API key missing"}
        try:
            client = initialize_openai_client(openai_api_key, openai_base_url, openai_timeout)
            bio_args = argparse.Namespace(gamma=gamma, delay=delay_between_calls, verbose=verbose)
            return evaluate_bio_jsonl(answer_data_list, golden_data_list, client, bio_args)
        except Exception as e:
            print(f"Error during BIO evaluation setup: {e}")
            return {"factscore": 0.0, "error": str(e)}
    else:
        print(f"Error: Dataset type '{dataset_type}' JSONL evaluation not implemented or misconfigured.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估JSONL格式的问答数据集")
    parser.add_argument('--golden_file', type=str, required=True, help="Path to the golden answers JSONL file.")
    parser.add_argument('--answer_file', type=str, required=True, help="Path to the generated answers JSONL file.")
    parser.add_argument('--dataset', type=str, required=True, choices=['POPQA', 'ASQA', 'BIO', 'ARC_Challenge'], help="Type of dataset to evaluate.")
    parser.add_argument('--num_samples', type=int, default=-1, help="Number of samples to evaluate (-1 for all).")
    
    parser.add_argument('--openai_api_key', type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key. Defaults to OPENAI_API_KEY env var.")
    parser.add_argument('--openai_base_url', type=str, default=os.environ.get("OPENAI_BASE_URL"), help="Custom OpenAI base URL (for proxies). Defaults to OPENAI_BASE_URL env var.")
    parser.add_argument('--openai_timeout', type=int, default=60, help="Timeout for OpenAI API calls in seconds.")
    
    parser.add_argument('--gamma', type=int, default=10, help="Gamma parameter (contextual, less relevant for FActScore precision).")
    parser.add_argument('--delay', type=float, default=1.0, help="Delay in seconds between OpenAI API calls for FactScore.")
    parser.add_argument('--verbose', action='store_true', help="Enable detailed output, especially for FactScore.")
    
    args = parser.parse_args()

    if args.verbose: print("启用详细输出模式")

    if args.dataset == 'BIO' and not args.openai_api_key:
        args.openai_api_key = "sk-x6RV5LxPPAMQiD97TpLU8NA2hB3iYesjJPzsBy1FVW0NS1uy" 
        print(f"Warning: OpenAI API key not provided via --openai_api_key or ENV. Using hardcoded fallback for BIO. This is not recommended for production.")
        if not args.openai_base_url:
             args.openai_base_url = "https://api.openai-proxy.org/v1"
             print(f"Warning: OpenAI Base URL not set. Using hardcoded proxy for BIO: {args.openai_base_url}")

    results = evaluate_file_jsonl(
        args.golden_file, args.answer_file, args.dataset, args.num_samples,
        args.openai_api_key, args.openai_base_url, args.openai_timeout,
        args.gamma, args.delay, args.verbose
    )

    if results:
        print("\n" + "="*25)
        print("  Final Evaluation Summary")
        print("="*25)
        print(f"Dataset      : {args.dataset}")
        print(f"Answer File  : {os.path.basename(args.answer_file)}")
        print("-"*25)
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric:<22}: {value:.4f}")
            else:
                print(f"  {metric:<22}: {value}")
        print("="*25 + "\n")
    else:
        print("\nEvaluation did not produce results or encountered an error.")
