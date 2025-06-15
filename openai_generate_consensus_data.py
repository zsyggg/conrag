import os
import random
import time
import json
from openai import OpenAI
from tqdm import tqdm

# --- 配置 OpenAI API ---
# 从环境变量读取API密钥和基础URL，如果未设置，则使用您提供的值作为备用
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-x6RV5LxPPAMQiD97TpLU8NA2hB3iYesjJPzsBy1FVW0NS1uy")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1") # 注意：官方API不需要 /v1 后缀，代理可能需要
OPENAI_MODEL_NAME = "gpt-4o-2024-05-13" # 您可以根据实际情况调整

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# --- 文件处理函数 ---

def read_jsonl_data(file_path):
    """
    从 JSONL 文件读取数据。
    每行是一个 JSON 对象，期望包含 "query" (string) 和 "passages" (list of strings)。
    """
    data_to_process = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                if "query" in record and "passages" in record:
                    if isinstance(record["query"], str) and isinstance(record["passages"], list):
                        # 确保passages列表中的所有元素都是字符串
                        record["passages"] = [str(p) for p in record["passages"]]
                        data_to_process.append(record)
                    else:
                        print(f"Warning: Line {line_number}: Invalid data types for query or passages. Query should be str, passages list of str.")
                else:
                    print(f"Warning: Line {line_number}: Missing 'query' or 'passages' field in record: {line.strip()}")
            except json.JSONDecodeError:
                print(f"Warning: Line {line_number}: Could not decode JSON: {line.strip()}")
    return data_to_process


def write_consensus_to_file(file_path, query, documents, consensus_text):
    """
    将生成的共识写入文件，格式与 train_consensus.py 的输入一致。
    query: string
    documents: list of strings
    consensus_text: string
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"query: {query}\n")
            # 将文档列表转换为JSON字符串列表的格式，以匹配 train_consensus.py 的解析逻辑
            documents_str = json.dumps(documents, ensure_ascii=False)
            f.write(f"documents: {documents_str}\n")
            f.write(f"consensus: {consensus_text}\n\n")
    except Exception as e:
        print(f"Error writing to file: {e}")

# --- OpenAI API 调用函数 ---

def generate_consensus_with_openai(query, documents, retries=3, delay=5):
    """
    使用 OpenAI API 根据问题和文档列表生成共识。
    """
    if not documents: # 如果文档列表为空，则不生成
        print(f"Warning: No documents provided for query: {query}. Skipping consensus generation.")
        return None

    # 构建文档的字符串表示形式，每个文档换行
    documents_str = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])

    prompt = f"""
Please act as an expert summarizer. Based on the following query and the provided documents, generate a concise, factual, and comprehensive consensus answer.
The consensus should synthesize information from the documents to directly answer the query.
Avoid personal opinions or information not present in the documents.

Query:
{query}

Documents:
{documents_str}

Consensus Answer:
"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert summarizer tasked with generating a consensus answer from provided documents based on a query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2, # 低温以获得更具事实性的输出
                max_tokens=300, # 根据需要调整共识的最大长度 (略微增加)
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            consensus_text = response.choices[0].message.content.strip()
            # 简单的后处理，移除可能的多余引言
            if consensus_text.lower().startswith("consensus answer:"):
                consensus_text = consensus_text[len("consensus answer:"):].strip()
            return consensus_text
        except Exception as e:
            print(f"OpenAI API call failed on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to generate consensus for query after {retries} attempts: {query}")
                return None

# --- 主函数 ---

def main():
    # --- 配置路径 ---
    # 输入文件：包含检索结果的JSONL文件
    input_file_path = '/workspace/conRAG/data/popqa/popqa_retrieved.jsonl'
    # 输出文件：用于训练T5共识器的数据，格式为 query:, documents:, consensus:
    # 为区分，可以在文件名中指明数量
    output_file_path = '/workspace/conRAG/data/popqa/popqa_consensus_for_t5_training_gpt4o_500_samples.txt'
    # 要处理的记录数量
    num_records_to_process = 500

    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        print("Please ensure the retrieved data file exists at the specified path.")
        return

    print(f"Reading data from: {input_file_path}")
    all_data = read_jsonl_data(input_file_path)

    if not all_data:
        print(f"No data successfully read from {input_file_path}. Please check the file content and format.")
        return

    # 选择要处理的数据子集
    data_to_process = all_data[:num_records_to_process]
    actual_records_to_process = len(data_to_process) # 实际将处理的记录数

    if actual_records_to_process == 0:
        print("No records to process (either original file was empty or num_records_to_process is 0).")
        return

    print(f"Found {len(all_data)} total items. Will process the first {actual_records_to_process} items.")

    # 清空或创建输出文件，以便重新写入
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("") # 创建或清空文件
    print(f"Output will be saved to: {output_file_path}")

    for item in tqdm(data_to_process, desc=f"Generating Consensus for {actual_records_to_process} items with GPT-4o"):
        query = item["query"]
        documents = item["passages"] # "passages" 字段现在直接是文档列表

        # print(f"\nProcessing query: {query}") # tqdm 已经提供了进度，这行可以注释掉以减少输出

        consensus_text = generate_consensus_with_openai(query, documents)

        if consensus_text:
            # print(f"Generated Consensus (first 100 chars): {consensus_text[:100]}...") # 可以注释掉以减少输出
            write_consensus_to_file(output_file_path, query, documents, consensus_text)
        else:
            print(f"Failed to generate consensus for query: {query}") # 失败时仍然打印
            # 即使失败，也写入一个标记，方便后续检查和处理
            write_consensus_to_file(output_file_path, query, documents, "CONSENSUS_GENERATION_FAILED")

    print(f"\nConsensus generation complete for {actual_records_to_process} items. Output saved to: {output_file_path}")
    print(f"Please use '{output_file_path}' as the --train_file for your train_consensus.py script.")

if __name__ == '__main__':
    main()
