# utils.py
import re
from typing import Dict, List, Any

def format_arc_choices_for_prompt(choices_data: Dict) -> str:
    """
    格式化选项用于prompt，每个选项换行
    输入: {"text": [...], "label": [...]}
    输出: "\nA: text1\nB: text2\nC: text3\nD: text4"
    """
    if not choices_data:
        return ""
    
    if isinstance(choices_data, dict) and 'text' in choices_data and 'label' in choices_data:
        formatted = ""
        for label, text in zip(choices_data['label'], choices_data['text']):
            formatted += f"\n{label}: {text}"
        return formatted
    
    return ""

def postprocess_arc_answer(answer: str) -> str:
    """
    后处理ARC答案，提取选项字母
    """
    # 清理答案
    answer = answer.strip()
    
    # 尝试多种模式提取答案
    patterns = [
        r'^([A-E])(?:[.)\s:]|$)',  # 开头的A. 或 A) 或 A: 或 就是A
        r'answer is ([A-E])',
        r'correct answer is ([A-E])',
        r'choose ([A-E])',
        r'\b([A-E])\b'  # 独立的字母
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 如果答案就是单个字母
    if len(answer) == 1 and answer.upper() in "ABCDE":
        return answer.upper()
    
    # 返回原始答案的前10个字符（用于调试）
    return answer[:10] if len(answer) > 10 else answer