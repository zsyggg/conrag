from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 Qwen2-7B-Instruct 模型
model_path = "/home/zhangshuaiyu/.cache/modelscope/hub/qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 读取文件内容并处理
def read_lines_from_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        # 假设每个问题有10行对应的文档
        num_lines_per_query = 10

        for i in range(0, len(lines), num_lines_per_query):
            query, documents = None, []
            for j in range(num_lines_per_query):
                line = lines[i + j].strip()
                if line:
                    parts = line.split("[SEP]")
                    if len(parts) == 2:
                        if query is None:
                            query = parts[0].strip()  # 只提取一次问题
                        documents.append(parts[1].strip())  # 只提取文档内容
            if query and documents:
                data.append((query, documents))
    
    return data

# 写入生成的共识到文件
def write_results_to_file(file_path, query, documents, consensus):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"query: {query}\n")
        f.write(f"documents: {documents}\n")
        f.write(f"consensus: {consensus}\n\n")

# 使用 Qwen2-7B-Instruct 生成共识
def send_request(query, documents):
    combined_documents = '", "'.join(documents)
    combined_documents = f'["{combined_documents}"]'  # 将文档内容放在双引号分隔的列表中
    prompt = f"Please generate a concise consensus answer based on the following question and documents:\nQuestion: {query}\nDocuments: {combined_documents}\nConsensus:"

    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=150, do_sample=True, temperature=1.0)
    consensus = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 解析模型生成的文本，提取出"Consensus:"之后的部分
    consensus_text = consensus.split("Consensus:")[-1].strip()
    consensus_list = [consensus_text]  # 将共识放在列表中
    
    return consensus_list

# 主函数
def main():
    input_file_path = '/home/zhangshuaiyu/CRAG/data/popqa/train_popqa.txt'
    output_file_path = './local_consensus.txt'

    data = read_lines_from_file(input_file_path)

    for query, documents in data:
        result = send_request(query, documents)
        if result:  # 只有在生成了内容时才写入文件
            print(f"Generated Consensus: {result}")
            write_results_to_file(output_file_path, query, documents, result)

if __name__ == '__main__':
    main()
