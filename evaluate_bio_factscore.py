# eval_bio_fixed.py
import json
import os
import subprocess
import tempfile
from typing import List, Dict, Tuple
import re

class BIOFactScoreEvaluator:
    """修正后的BIO FactScore评估器"""
    
    def __init__(self, openai_key: str, cache_dir: str = "./factscore_cache"):
        self.openai_key = openai_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def format_for_factscore(self, predictions: List[Dict]) -> str:
        """将预测结果格式化为FactScore需要的格式"""
        formatted_data = []
        
        for pred in predictions:
            # 提取人物名称
            query = pred.get('query', '')
            if query.startswith('Tell me a bio of '):
                person_name = query.replace('Tell me a bio of ', '').strip('.')
            else:
                # 尝试其他格式
                person_name = query.strip()
            
            # 获取生成的传记
            generated_bio = pred.get('generated_answer', '')
            
            # 跳过无效数据
            if not person_name or not generated_bio:
                continue
            if 'ERROR' in generated_bio or 'SKIPPED' in generated_bio:
                continue
                
            formatted_item = {
                'topic': person_name,
                'output': generated_bio
            }
            formatted_data.append(formatted_item)
        
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')
            return f.name
    
    def run_factscore(self, formatted_path: str) -> Dict:
        """运行官方FactScore评估"""
        cmd = [
            'python', '-m', 'factscore.factscorer',
            '--data_path', formatted_path,
            '--model_name', 'retrieval+ChatGPT',
            '--cache_dir', self.cache_dir,
            '--openai_key', self.openai_key,
            '--verbose'
        ]
        
        print(f"Running FactScore evaluation...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 解析输出
            output = result.stdout
            print("FactScore Output:")
            print(output)
            
            # 提取分数
            score_match = re.search(r'FActScore:\s*([0-9.]+)', output)
            if score_match:
                score = float(score_match.group(1))
                # 如果是百分比，转换为小数
                if score > 1:
                    score = score / 100
                return {
                    'factscore': score,
                    'output': output
                }
            else:
                # 尝试其他格式
                json_match = re.search(r'\{.*"score":\s*([0-9.]+).*\}', output)
                if json_match:
                    score = float(json_match.group(1))
                    return {
                        'factscore': score,
                        'output': output
                    }
                
        except subprocess.CalledProcessError as e:
            print(f"Error running FactScore: {e}")
            print(f"Stderr: {e.stderr}")
            
        return {
            'factscore': 0.0,
            'error': 'Failed to extract score',
            'output': output if 'output' in locals() else ''
        }
    
    def evaluate(self, predictions_file: str, num_samples: int = -1) -> Dict:
        """主评估函数"""
        # 加载预测结果
        predictions = []
        with open(predictions_file, 'r') as f:
            for i, line in enumerate(f):
                if num_samples > 0 and i >= num_samples:
                    break
                predictions.append(json.loads(line))
        
        print(f"Loaded {len(predictions)} predictions")
        
        # 格式化数据
        formatted_path = self.format_for_factscore(predictions)
        print(f"Formatted data saved to: {formatted_path}")
        
        # 运行评估
        results = self.run_factscore(formatted_path)
        
        # 清理临时文件
        os.unlink(formatted_path)
        
        return results


def evaluate_bio_jsonl_corrected(predictions_file: str,
                                 openai_api_key: str,
                                 num_samples: int = -1) -> Dict:
    """
    修正后的BIO评估函数，直接调用官方FactScore
    
    Args:
        predictions_file: 模型输出的JSONL文件路径
        openai_api_key: OpenAI API密钥
        num_samples: 要评估的样本数量，-1表示全部
    
    Returns:
        包含factscore的字典
    """
    evaluator = BIOFactScoreEvaluator(openai_api_key)
    return evaluator.evaluate(predictions_file, num_samples)


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='Path to predictions JSONL')
    parser.add_argument('--openai_key', required=True, help='OpenAI API key')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to evaluate')
    args = parser.parse_args()
    
    results = evaluate_bio_jsonl_corrected(
        args.predictions,
        args.openai_key,
        args.num_samples
    )
    
    print(f"\nFinal FactScore: {results['factscore']:.4f}")
