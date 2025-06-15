import argparse
import json
import os
import re
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

def regex_split_into_sentences(text):
    """
    使用正则表达式将文本分割成句子。
    这是一个比NLTK更简单的替代方案。
    """
    if not text or not isinstance(text, str):
        return []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def generate_t5_consensus(model, tokenizer, query, passages_list, device, max_length=256):
    """使用T5模型生成共识文本。"""
    if not passages_list:
        return "NO_PASSAGES_PROVIDED_TO_T5"
    passages_concatenated = " ".join(filter(None, passages_list))
    if not passages_concatenated.strip():
        return "PASSAGES_EMPTY_AFTER_FILTERING"
    input_text = f"query: {query} documents: {passages_concatenated}"
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    generation_args = {
        "max_length": max_length, "num_beams": 4, "early_stopping": True,
        "repetition_penalty": 1.2, "length_penalty": 1.5,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)
    consensus_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return consensus_text

def find_additional_evidence(
    st_model,
    query,
    consensus_text,
    all_doc_sentences,
    top_n=2,
    penalty_lambda=0.7,
    redundancy_threshold=0.8,
):
    """从文档中筛选与查询强相关且与共识正交的句子。

    Args:
        st_model: SentenceTransformer 模型。
        query (str): 用户问题。
        consensus_text (str): 生成的共识摘要。
        all_doc_sentences (list[str]): 候选句子列表。
        top_n (int): 返回的证据数量。
        penalty_lambda (float): 与共识相似度的惩罚权重。
        redundancy_threshold (float): 证据之间相似度超过该阈值时视为冗余。

    Returns:
        list[str]: 额外证据句子列表。
    """

    if not all_doc_sentences:
        return []

    # --- 计算嵌入 ---
    query_emb = st_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    consensus_sentences = regex_split_into_sentences(consensus_text)
    if consensus_sentences:
        consensus_embs = st_model.encode(consensus_sentences, convert_to_tensor=True, show_progress_bar=False)
    else:
        consensus_embs = None
    candidate_embs = st_model.encode(all_doc_sentences, convert_to_tensor=True, show_progress_bar=False)

    # --- 计算与查询的相关性 ---
    query_sims = util.pytorch_cos_sim(candidate_embs, query_emb).squeeze()

    # --- 计算与共识的最大相似度 ---
    if consensus_embs is not None:
        # 对每个候选句子，取其与所有共识句子的最大相似度
        consensus_sims = util.pytorch_cos_sim(candidate_embs, consensus_embs).max(dim=1).values
    else:
        consensus_sims = torch.zeros(len(all_doc_sentences), device=candidate_embs.device)

    # --- 综合得分 ---
    scores = query_sims - penalty_lambda * consensus_sims

    # --- 根据得分排序 ---
    ranked_indices = torch.argsort(scores, descending=True)

    selected_sentences = []
    selected_embs = []
    for idx in ranked_indices:
        sentence = all_doc_sentences[idx]
        if sentence.strip() == "":
            continue
        # 与共识完全相同则跳过
        if consensus_sentences and sentence.strip().lower() in (s.lower() for s in consensus_sentences):
            continue

        emb = candidate_embs[idx]

        # 与已选择证据的相似度检查，确保正交
        if selected_embs:
            redundancy = util.pytorch_cos_sim(emb, torch.stack(selected_embs)).max().item()
            if redundancy >= redundancy_threshold:
                continue

        selected_sentences.append(sentence)
        selected_embs.append(emb)
        if len(selected_sentences) >= top_n:
            break

    return selected_sentences


def main():
    parser = argparse.ArgumentParser(description="Generate consensus and extract additional orthogonal evidence (Optimized Version).")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--consensus_model_path', type=str, required=True)
    parser.add_argument('--st_model_name', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--top_n_orthogonal', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--max_consensus_length', type=int, default=256)
    parser.add_argument('--orthogonality_penalty', type=float, default=0.7, help="Penalty weight for consensus similarity.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    device = torch.device(args.device)

    print(f"Loading T5 consensus model from: {args.consensus_model_path}")
    t5_tokenizer = T5Tokenizer.from_pretrained(args.consensus_model_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(args.consensus_model_path).to(device)
    t5_model.eval()

    print(f"Loading SentenceTransformer model: {args.st_model_name}")
    try:
        st_model = SentenceTransformer(args.st_model_name, device=device)
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        exit(1)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    output_records = []
    with open(args.input_file, 'r', encoding='utf-8') as infile:
        all_lines = infile.readlines()
        for line_idx, line in enumerate(tqdm(all_lines, desc="Processing queries (Optimized Logic)")):
            data = None
            query = "Unknown query"
            try:
                data = json.loads(line.strip())
                query = data.get('query', 'QueryFieldMissing')
                original_passages = data.get('passages', [])

                # 跳过没有有效段落的记录
                valid_passages = [p for p in original_passages if isinstance(p, str) and p.strip()]
                if not valid_passages:
                    output_records.append({ "query": query, "original_passages": original_passages, "consensus": "NO_VALID_PASSAGES_PROVIDED", "additional_evidence": [] })
                    continue

                # 1. 生成共识
                consensus_text = generate_t5_consensus(t5_model, t5_tokenizer, query, valid_passages, device, args.max_consensus_length)
                
                # 2. 从所有段落中提取所有句子
                all_doc_sentences_flat = []
                for doc_text in valid_passages:
                    all_doc_sentences_flat.extend(regex_split_into_sentences(doc_text))
                
                # 3. 使用新的优化逻辑寻找额外证据
                additional_evidence = find_additional_evidence(
                    st_model,
                    query,
                    consensus_text,
                    list(set(all_doc_sentences_flat)), # 使用set去重
                    top_n=args.top_n_orthogonal,
                    penalty_lambda=args.orthogonality_penalty
                )
                
                output_records.append({
                    "query": query, "original_passages": original_passages,
                    "consensus": consensus_text, "additional_evidence": additional_evidence
                })

            except Exception as e:
                print(f"An error occurred processing line {line_idx+1} (Query: '{str(query)[:50]}...'). Error: {type(e).__name__} - {e}")
                current_passages = data.get('passages', []) if data else [line.strip()]
                output_records.append({ "query": query, "original_passages": current_passages, "consensus": f"ERROR_PROCESSING_ITEM: {type(e).__name__}", "additional_evidence": [] })

    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for record in output_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Optimized consensus generation complete. Output saved to {args.output_file}")

if __name__ == '__main__':
    main()
