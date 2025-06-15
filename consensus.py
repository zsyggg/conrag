import json
import argparse
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_consensus(consensus_model, tokenizer, query, passages, device):
    # 将 query 和所有 passages 合并为输入
    input_text = f"query: {query} documents: {passages}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # 调整生成参数以生成更丰富的内容
    generation_args = {
        "max_length": 256,
        "num_beams": 4,
        "early_stopping": False,
        "repetition_penalty": 1.1,
        "length_penalty": 2.0,
        "num_return_sequences": 3,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95
    }
    
    with torch.no_grad():
        outputs = consensus_model.generate(**inputs, **generation_args)
    
    consensus_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    return max(consensus_texts, key=len)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the cleaned retrieved JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the generated consensus TXT file')
    parser.add_argument('--consensus_model_path', type=str, required=True, help='Path to the finetuned consensus model')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载共识模型和 tokenizer
    consensus_tokenizer = T5Tokenizer.from_pretrained(args.consensus_model_path)
    consensus_model = T5ForConditionalGeneration.from_pretrained(args.consensus_model_path).to(device)

    with open(args.input_file, 'r', encoding='utf-8') as infile, open(args.output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, desc="Generating Consensus"):
            data = json.loads(line.strip())
            query = data['query']

            # 检查并获取文档内容
            if 'retrieved_docs' in data:
                passages = [doc['text'] for doc in data['retrieved_docs']]
            elif 'passages' in data:
                passages = [doc['text'] for doc in data['passages']]
            else:
                print(f"Skipping entry without 'retrieved_docs' or 'passages': {line}")
                continue  # 跳过没有文档的条目

            # 生成共识
            consensus = generate_consensus(consensus_model, consensus_tokenizer, query, " ".join(passages), device)
            
            # 将共识输出到文件中，documents以列表形式
            documents_str = json.dumps(passages, ensure_ascii=False)  # 确保以JSON格式输出列表
            output_text = f"query: {query}\ndocuments: {documents_str}\nconsensus: {consensus}\n\n"
            outfile.write(output_text)
    
    print(f"Consensus saved to {args.output_file}")

if __name__ == '__main__':
    main()
