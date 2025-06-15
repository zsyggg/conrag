# merge_arc_choices.py
import json
from tqdm import tqdm

def ensure_arc_data_has_choices(enhanced_file, original_file_with_choices, output_file):
    """确保增强数据包含选项"""
    
    # 读取有选项的原始数据
    print(f"Reading original data with choices from: {original_file_with_choices}")
    choices_map = {}
    with open(original_file_with_choices, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            question = data.get('question', '').strip()
            choices_map[question] = {
                'choices': data.get('choices'),
                'answerKey': data.get('answerKey')
            }
    
    print(f"Loaded {len(choices_map)} questions with choices")
    
    # 更新增强数据
    print(f"Reading enhanced data from: {enhanced_file}")
    updated_data = []
    matched = 0
    
    with open(enhanced_file, 'r') as f:
        for line in tqdm(f, desc="Merging choices"):
            data = json.loads(line)
            query = data.get('query', '').strip()
            
            # 查找匹配的选项
            if query in choices_map:
                data['choices'] = choices_map[query]['choices']
                data['answerKey'] = choices_map[query]['answerKey']
                matched += 1
            else:
                # 尝试模糊匹配
                for orig_q, choice_info in choices_map.items():
                    if query.lower() in orig_q.lower() or orig_q.lower() in query.lower():
                        data['choices'] = choice_info['choices']
                        data['answerKey'] = choice_info['answerKey']
                        matched += 1
                        break
            
            updated_data.append(data)
    
    # 写入新文件
    with open(output_file, 'w') as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Matched {matched}/{len(updated_data)} items with choices")
    print(f"Output saved to: {output_file}")

# 使用
if __name__ == "__main__":
    ensure_arc_data_has_choices(
        '/workspace/conRAG/data/arc_challenge/arc_challenge_enhanced_consensus_evidence.jsonl',
        '/workspace/conRAG/eval_data/arc_challenge_processed.jsonl',
        '/workspace/conRAG/data/arc_challenge/arc_challenge_enhanced_final.jsonl'
    )