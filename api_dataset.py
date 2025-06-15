import random
from http import HTTPStatus
import dashscope

# 替换为你的API密钥
API_KEY = 'sk-6acce737d4dc486fbf867f8dcb252282'
dashscope.api_key = API_KEY

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
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"query: {query}\n")
            f.write(f"documents: {documents}\n")
            f.write(f"consensus: {consensus}\n\n")
    except Exception as e:
        print(f"Failed to write results to file: {e}")

# 使用 Qwen2-7B API 生成共识
def send_request(query, documents):
    combined_documents = '", "'.join(documents)
    combined_documents = f'["{combined_documents}"]'  # 将文档内容放在双引号分隔的列表中
    prompt = f"Please generate a concise consensus answer based on the following question and documents:\nQuestion: {query}\nDocuments: {combined_documents}\nConsensus:"

    messages = [{'role': 'user', 'content': prompt}]
    
    try:
        response = dashscope.Generation.call(
            'qwen2-7b-instruct',
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message',
            stream=False,  # 关闭流模式以一次性获取完整响应
            output_in_full=True
        )

        # 如果响应是字符串，直接处理并返回
        if isinstance(response, str):
            return [response]
        
        # 处理API响应
        if response.status_code == HTTPStatus.OK:
            full_content = response.output.choices[0]['message']['content']
            consensus_text = full_content.split("Consensus:")[-1].strip()
            return [consensus_text]
        else:
            print(f"Request id: {response.request_id}, Status code: {response.status_code}, "
                  f"error code: {response.code}, error message: {response.message}")
            return None
    except Exception as e:
        print(f"Error encountered: {e}. Skipping this request.")
        return None

# 主函数
def main():
    input_file_path = '/home/zhangshuaiyu/CRAG/data/popqa/train_popqa.txt'
    output_file_path = './pop_con.txt'

    data = read_lines_from_file(input_file_path)

    for query, documents in data:
        print(f"Processing query: {query}")
        result = send_request(query, documents)
        
        if result:  # 只有在生成了内容时才写入文件
            print(f"Generated Consensus: {result}")
            write_results_to_file(output_file_path, query, documents, result)
        else:
            print(f"No consensus generated for query: {query}")

if __name__ == '__main__':
    main()
