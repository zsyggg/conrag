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

def find_additional_evidence(st_model, query, consensus_text, all_doc_sentences, top_n=2, penalty_lambda=0.7):
    """
    寻找额外证据的核心逻辑优化版本。

    Args:
        st_model: SentenceTransformer 模型。
        query (str): 用户问题。
        consensus_text (str): T5 生成的共识。
        all_doc_sentences (list of str): 从所有文档中提取的句子列表。
        top_n (int): 需要返回的证据数量。
        penalty_lambda (float): 对与共识相似的句子的惩罚权重。

    Returns:
        list of str: 额外的证据句子列表。
    """
    if not all_doc_sentences:
        return []

    # 1. 计算所有句子相对于 query 和 consensus 的嵌入
    query_embedding = st_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    consensus_embedding = st_model.encode(consensus_text, convert_to_tensor=True, show_progress_bar=False)
    doc_embeddings = st_model.encode(all_doc_sentences, convert_to_tensor=True, show_progress_bar=False)

    # 2. 计算相似度得分
    # (N, 1) 张量，其中 N 是文档句子的数量
    query_similarities = util.pytorch_cos_sim(doc_embeddings, query_embedding)
    # (N, 1) 张量
    consensus_similarities = util.pytorch_cos_sim(doc_embeddings, consensus_embedding)

    # 3. 计算最终得分
    # 核心思想：最终得分 = 与问题的相关性 - 对与共识相似性的惩罚
    # 我们希望与问题相关度高，与共识相似度低的句子得分更高
    final_scores = query_similarities - penalty_lambda * consensus_similarities
    
    # 4. 排序并选出 Top N
    # 将得分展平以便排序
    final_scores_flat = final_scores.flatten()
    # 获取得分最高的 top_n+1 个索引（多选一个以防最相关的就是共识本身）
    top_indices = torch.topk(final_scores_flat, k=min(top_n + 1, len(all_doc_sentences)), sorted=True).indices

    # 5. 组装证据，并进行简单的冗余检查
    additional_evidence = []
    consensus_norm = consensus_text.strip().lower()
    for index in top_indices:
        sentence = all_doc_sentences[index.item()]
        # 避免将与共识完全相同的句子作为证据
        if sentence.strip().lower() != consensus_norm and sentence not in additional_evidence:
            additional_evidence.append(sentence)
        if len(additional_evidence) >= top_n:
            break
            
    return additional_evidence


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
