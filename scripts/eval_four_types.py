import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional

from datasets import load_dataset
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from mathruler.grader import extract_boxed_content, grade_answer


# ===================================================================
# === 部分 1: 评测逻辑 (来自您提供的第二个代码块) ===
# ===================================================================

# --- 评测脚本中的常量和函数 ---
REFUSAL_MISSING_PHRASE = "refused to answer, because the image is missing"
REFUSAL_IRRELEVANT_PHRASE = "refused to answer, because the image is redundant and irrelevant"

def is_correct_refusal(
    response: str,
    refusal_type: Literal["missing", "irrelevant"],
    allow_flexible_matching: bool = True
) -> bool:
    """检查响应是否为正确且直接的拒答。"""
    if not response:
        return False
    normalized_response = response.strip().lower().rstrip(".")

    if refusal_type == "missing":
        target_phrase = REFUSAL_MISSING_PHRASE
    elif refusal_type == "irrelevant":
        target_phrase = REFUSAL_IRRELEVANT_PHRASE
    else:
        return False

    if normalized_response == target_phrase:
        return True

    if allow_flexible_matching:
        # if target_phrase in normalized_response and len(normalized_response) < len(target_phrase) + 20:
        #     return True
        
        if refusal_type == "missing":
            pattern = re.compile(r"refused.*answer.*image.*missing", re.IGNORECASE | re.DOTALL)
        else:
            pattern = re.compile(r"refused.*answer.*image.*(irrelevant|redundant)", re.IGNORECASE | re.DOTALL)
        
        if re.search(pattern, response):
            return True

    return False

def has_reasoning_leakage(response: str) -> bool:
    """检查响应是否包含不应出现的推理或格式。"""
    return "<think>" in response or "</think>" in response or "\\boxed" in response

def format_reward(response: str) -> float:
    """检查标准的 <think>...</think>\\boxed{...} 格式。"""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    # 使用 re.search 而非 re.fullmatch 以允许尾部有额外内容
    format_match = re.search(pattern, response)
    return 1.0 if format_match else 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    """计算可解决问题的准确性。"""
    # 如果模型拒答了一个可解决的问题，则惩罚
    # if "refused to answer" in response.lower():
    #     return 0.0
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.0) -> List[Dict[str, float]]:
    """根据新的 prompt 逻辑计算奖励分数。"""
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input.get("response") or ""
        # 移除标签周围的空格以进行稳健匹配
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
        rollout_type = reward_input.get("rollout_type")

        format_score = 0.0
        accuracy_score = 0.0
        
        # 类别 1.2: 需要图片但图片缺失
        if rollout_type == "without_image":
            if is_correct_refusal(response, "missing"):# and not has_reasoning_leakage(response):
                accuracy_score = 1.0
                format_score = 1.0
        
        # 类别 1.3: 提供了无关图片
        elif rollout_type == "text_with_image":
            if is_correct_refusal(response, "irrelevant"):# and not has_reasoning_leakage(response):
                accuracy_score = 1.0
                format_score = 1.0
        
        # 类别 1.1 和 1.4: 标准可解问题
        else: # rollout_type is 'with_image' or 'text_without_image'
            # 对于标准问题，如果它拒答了，准确率应为0
            if False: #is_correct_refusal(response, "missing") or is_correct_refusal(response, "irrelevant"):
                 accuracy_score = 0.0
                 format_score = 0.0 # 格式也不对
            else:
                format_score = format_reward(response)
                accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        # 这里我们只关心最终的准确率，所以不考虑格式权重
        final_score = accuracy_score

        scores.append(
            {
                "overall": final_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores


# ===================================================================
# === 部分 2: 模型推理与主流程 (来自您提供的第一个代码块并修改) ===
# ===================================================================

# --- vLLM 服务器配置 ---
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8007/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
MODEL_NAME = "Qwen2-VL-7B-Instruct" # 模型名称

# --- Prompt 模板 ---
PROMPT_TEMPLATE = """{problem_text} Your task is to analyze and solve the provided math problem. Please follow these rules in strict order:

Rule 1: Check for a missing image.
If the image information that is **essential** for solving the problem is missing, your **only output must be** the following sentence and nothing else:
Refused to answer, because the image is missing

Rule 2: Check for an irrelevant image.
If the provided image is **completely irrelevant** to the problem's content, your **only output must be** the following sentence and nothing else:
Refused to answer, because the image is redundant and irrelevant

Rule 3: Solve the problem normally.
Only if the image is present and relevant to the problem, execute this rule:
First, present your reasoning process as an internal monologue, enclosed entirely within <think> </think> tags. Then, put your final answer in \boxed{{}}.
"""

# --- 数据加载与四类样本生成 ---
print("Loading and preparing dataset...")
# 注意：请确保数据集路径正确
dataset = load_dataset('/mnt/siqingyi/codes/EasyR1_ours/data/test_mix', split='test')

all_samples_to_process = []
for i, item in enumerate(tqdm(dataset, desc="Generating four sample types")):
    base_item = {
        "id": i,
        "problem": item["problem"],
        "answer": str(item["answer"]),
        "image": item["images"][0] if item.get("images") else None,
    }

    # i 为偶数: 真实多模态样本 (x_i 的 image 为真实图)
    if i % 2 == 0:
        # 1.1 with image: 多模态样本
        all_samples_to_process.append({**base_item, "rollout_type": "with_image", "use_image": True})
        # 1.2 without image: 多模态样本去掉图片
        all_samples_to_process.append({**base_item, "rollout_type": "without_image", "use_image": False})
    # i 为奇数: 纯文本样本 + 虚假图 (x_i 的 image 为随机匹配的虚假图)
    else:
        # 1.3 text with image: 纯文本样本添加图片
        all_samples_to_process.append({**base_item, "rollout_type": "text_with_image", "use_image": True})
        # 1.4 text without image: 纯文本样本
        all_samples_to_process.append({**base_item, "rollout_type": "text_without_image", "use_image": False})

print(f"Total samples to evaluate across 4 types: {len(all_samples_to_process)}")


# --- 推理函数 ---
def process_item(item: Dict, client_instance: OpenAI) -> Dict:
    """对单个样本进行推理，返回结果"""
    error_message = None
    response_text = "N/A"

    try:
        # 格式化 prompt
        final_prompt_text = PROMPT_TEMPLATE.format(problem_text=item['problem'])
        
        # 构建 messages
        messages = [{"role": "user", "content": []}]
        
        # 如果需要，添加图片
        if item["use_image"] and item.get("image"):
            pil_image = item["image"]
            buffered = io.BytesIO()
            pil_image.convert('RGB').save(buffered, format="PNG")
            encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_image_url = f"data:image/png;base64,{encoded_image_text}"
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": base64_image_url},
            })

        # 添加文本内容
        messages[0]["content"].append({"type": "text", "text": final_prompt_text})

        # 调用模型 API
        chat_response = client_instance.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        response_text = chat_response.choices[0].message.content

    except Exception as e:
        error_message = str(e)
        print(f"Error processing item ID {item.get('id', 'N/A')} ({item.get('rollout_type')}): {e}")

    return {
        "response": response_text,
        "original_item": item,
        "error": error_message,
    }

# --- 并发执行与评测 ---
concurrent_requests_batch_size = 20
all_results = []

print(f"Starting evaluation with {concurrent_requests_batch_size} concurrent requests...")

with ThreadPoolExecutor(max_workers=concurrent_requests_batch_size) as executor:
    future_to_item = {executor.submit(process_item, item, client): item for item in all_samples_to_process}

    for future in tqdm(as_completed(future_to_item), total=len(all_samples_to_process), desc="Processing samples"):
        try:
            result = future.result()
            all_results.append(result)
        except Exception as exc:
            item_identity = future_to_item[future]
            print(f"Item ID {item_identity.get('id', 'N/A')} generated an exception: {exc}")
            all_results.append({
                "response": "N/A (Exception caught)",
                "original_item": item_identity,
                "error": str(exc)
            })

# --- 汇总结果并计算分数 ---
print("\nEvaluation finished. Computing scores...")

reward_inputs = []
for result in all_results:
    item = result["original_item"]
    reward_inputs.append({
        "response": result["response"],
        "ground_truth": item["answer"],
        "rollout_type": item["rollout_type"],
    })

# 计算分数
scores = compute_score(reward_inputs)

# 按类别统计准确率和格式分
category_stats = {
    "with_image": {"correct": 0, "total": 0, "format_sum": 0.0},
    "without_image": {"correct": 0, "total": 0, "format_sum": 0.0},
    "text_with_image": {"correct": 0, "total": 0, "format_sum": 0.0},
    "text_without_image": {"correct": 0, "total": 0, "format_sum": 0.0},
}

for i, score_data in enumerate(scores):
    rollout_type = reward_inputs[i]["rollout_type"]
    if rollout_type in category_stats:
        stats = category_stats[rollout_type]
        stats["total"] += 1
        stats["format_sum"] += score_data["format"]
        if score_data["accuracy"] == 1.0:
            stats["correct"] += 1
            
# ===================================================================
# === 新增部分: 将推理结果保存为 JSON 文件 ===
# ===================================================================
print("\nSaving inference results to json file...")
output_data_to_save = []
for result in all_results:
    # 复制原始项目信息，但不包括 'image' 字段
    original_item_copy = {k: v for k, v in result["original_item"].items() if k != 'image'}
    
    output_data_to_save.append({
        "response": result["response"],
        "original_item": original_item_copy,
        "error": result["error"],
    })

output_filename = "inference_results_55_step_new.json"
try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data_to_save, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved results to {output_filename}")
except Exception as e:
    print(f"Error saving results to JSON file: {e}")


# --- 打印最终报告 ---
print("\n" + "="*65)
print("                    Model Evaluation Final Report")
print("="*65)

total_correct = 0
total_samples = 0
total_format_sum = 0.0

for category, stats in category_stats.items():
    total = stats["total"]
    correct = stats["correct"]
    format_sum = stats["format_sum"]
    
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_format_score = (format_sum / total * 100) if total > 0 else 0
    
    total_correct += correct
    total_samples += total
    total_format_sum += format_sum
    
    print(f"Category: {category:<20} | Accuracy: {accuracy:6.2f}% ({correct:>4}/{total:>4}) | Format Score: {avg_format_score:6.2f}%")

overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
overall_avg_format_score = (total_format_sum / total_samples * 100) if total_samples > 0 else 0
print("-" * 65)
print(f"Overall Accuracy: {overall_accuracy:6.2f}% ({total_correct}/{total_samples})")
print(f"Overall Format Score: {overall_avg_format_score:6.2f}%")
print("="*65)
