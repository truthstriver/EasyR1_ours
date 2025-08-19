import json
import argparse
import os
from tqdm import tqdm
from math import comb
import warnings

# ==============================================================================
# 核心要求：假设 utils 目录及以下模块和函数已经存在，并直接导入使用。
# 脚本本身不关心这些函数的内部实现。
# ==============================================================================
try:
    # 遵循第一个脚本的结构，我们假设答案提取函数在 utils.parser 中
    from utils.parser import extract_answer
    # 假设正确性检查函数在 utils.grader 中
    from utils.grader import check_is_correct
except ImportError:
    print("="*80)
    print("错误：无法从 'utils' 目录导入 'extract_answer' 或 'check_is_correct'。")
    print("请确保您的工作目录下存在一个 'utils' 文件夹，并且其中包含带有这些函数的 'parser.py' 和 'grader.py' 模块。")
    print("="*80)
    exit(1)


def parse_args():
    """解析命令行参数，与第一个脚本保持一致。"""
    parser = argparse.ArgumentParser(
        description="Evaluate model generation results using predefined utils."
    )
    parser.add_argument(
        '--generation_path', 
        type=str, 
        required=True, 
        help="Path to the JSON result file to be evaluated."
    )
    parser.add_argument(
        '--data_name', 
        type=str, 
        default="mathvision", 
        help='Dataset identifier, passed to extract_answer to select the correct parsing logic.'
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=1, 
        help="Value of k for pass@k calculation. Note: The input JSON likely has only one response per problem."
    )
    return parser.parse_args()


def load_json_results(filepath: str) -> list:
    """从指定的JSON文件中加载评测数据。"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"错误：文件未找到于 '{filepath}'")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'results' not in data or not isinstance(data['results'], list):
        raise ValueError("错误：JSON文件必须包含一个名为 'results' 的键，其值为一个列表。")
        
    return data['results']


def evaluate(args):
    """
    主评测函数。
    读取JSON文件，并使用从utils导入的函数进行评测。
    """
    # 1. 加载数据
    # file_outputs 的每一项都对应一个问题
    file_outputs = load_json_results(args.generation_path)
    
    print(f"已加载 {len(file_outputs)} 条结果，开始使用 'utils' 中的函数进行评测...")

    correct_cnt = 0
    total_cnt = len(file_outputs)
    pass_at_k_list = []
    k = args.k
    
    # 2. 遍历每一条结果进行评测，使用 tqdm 显示进度条
    for i, item in enumerate(tqdm(file_outputs, desc="评测中...")):
        
        # 从JSON项中获取标准答案和模型的原始输出
        gt_ans = item.get('ground_truth')
        # 确保我们总是评测未经处理的原始输出
        raw_response = item.get('model_output_raw')

        if gt_ans is None or raw_response is None:
            warnings.warn(f"跳过索引 {i}：缺少 'ground_truth' 或 'model_output_raw'。")
            total_cnt -= 1 # 从总数中移除，因为它无法被评测
            continue

        # 保持与第一个脚本的兼容性，将 response 放入一个列表中
        # 即使您第二个脚本的输出每个问题只有一个回答，这种结构也能处理
        generated_responses = [raw_response]
        
        # 3. **调用导入的 utils 函数**
        # a. 使用 extract_answer 提取答案
        generated_answers = [extract_answer(resp, args.data_name) for resp in generated_responses]
        
        # b. 使用 check_is_correct 检查答案是否正确
        is_correct_list = [check_is_correct(gen_ans, gt_ans) for gen_ans in generated_answers]
        
        # 4. 判断该问题是否至少有一个正确答案
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
            
        # 5. 计算 pass@k (逻辑与第一个脚本完全相同)
        n = len(generated_responses) # 采样的回答数量
        c = sum(is_correct_list)     # 正确的回答数量
        
        # 只有在采样数大于等于k时计算才有意义
        if n >= k:
            try:
                pass_at_k = 1.0 - (comb(n - c, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            except (ValueError, ZeroDivisionError):
                # 如果 c > n 或 n < k，comb可能会失败，这种情况下pass@k是1
                if n-c < k :
                    pass_at_k_list.append(1.0)
                else: # c=0, n>=k, 应该为0
                    pass_at_k_list.append(0.0)


    # 6. 打印最终结果，格式与第一个脚本一致
    print("\n评测完成！")
    print("--- 评测结果 ---")
    print(f"正确数 / 总数: {correct_cnt}/{total_cnt}")
    
    accuracy = correct_cnt / total_cnt if total_cnt > 0 else 0
    print(f"Acc: {accuracy:.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list) if pass_at_k_list else 0
        print(f"Pass@{k}: {average_pass_at_k:.4f}")
    elif total_cnt > 0:
        # 如果 pass_at_k_list 为空 (例如所有 n < k)，则 pass@1 就是准确率
        print(f"Pass@1: {accuracy:.4f}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)