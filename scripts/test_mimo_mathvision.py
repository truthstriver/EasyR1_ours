import base64
from openai import OpenAI
import os
import json
from tqdm import tqdm
import re
from datasets import load_dataset
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from mathruler.grader import extract_boxed_content, grade_answer

# # --- Helper functions (inspired by mathruler.grader) ---
# def extract_boxed_content(text: Optional[str]) -> Optional[str]:
#     if text is None:
#         return None
#     match = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return None

# def grade_answer(extracted_answer: Optional[str], ground_truth: str) -> bool:
#     if extracted_answer is None:
#         return False
#     return extracted_answer == ground_truth
# # --- End helper functions ---


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8006/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# ========================
# === 修改部分开始 ===
# ========================

# Load dataset (new path and format)
dataset = load_dataset('./data/MathVision')
test_data = dataset["train"]  # or "test" if split exists; adjust as needed

# Preprocess: decode images and extract fields
processed_test_data = []
for idx, item in enumerate(test_data):
    # Extract image (list of PIL images, we assume one image per item)
    images = item["images"]
    pil_image = images[0] if isinstance(images, list) and len(images) > 0 else None

    # Extract problem and answer
    problem_text = item["problem"].replace("<image>", "").strip()  # remove <image> placeholder
    answer = item["answer"]

    # Parse options if present in problem text
    # Example: problem has "Choices:\nA:40°\nB:60°..."
    options = []
    if "Choices:" in problem_text:
        choices_part = problem_text.split("Choices:")[1].strip().splitlines()
        options = [line.strip() for line in choices_part if line.strip()]

    # Construct question without options
    question = problem_text
    if "Choices:" in problem_text:
        question = problem_text.split("Choices:")[0].strip()

    processed_item = {
        "question": question,
        "options": options,
        "answer": answer,
        "decoded_image": pil_image,
        "id": idx  # use index as ID since not provided
    }
    processed_test_data.append(processed_item)

test_data = processed_test_data  # replace with processed format

# ========================
# === 修改部分结束 ===
# ========================

# Prepare output directory and file
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "MathVision_results_xiaomimimo_vl_mathruler_grading_6-23_ORIGIN.json")

# Batch evaluation
concurrent_requests_batch_size = 20
acc_dict = {"num": 0, "correct": 0}
results = []

print(f"Starting evaluation on {len(test_data)} items with {concurrent_requests_batch_size} concurrent requests...")

def process_item(item, client_instance):
    model_answer = None
    is_correct = False
    response_text_raw = "N/A"
    response_text_processed = "N/A"
    error_message = None

    problem_str = item["question"]
    if len(item["options"]) > 0:
        options_string_part = "Directly answer with letter in Options:\n" + "\n".join(item["options"])
        problem_str = item["question"] + "\n" + options_string_part

    item['problem_for_model'] = problem_str

    if 'decoded_image' not in item or not item['decoded_image']:
        return {
            "input_problem": item['problem_for_model'],
            "image_data_missing": True,
            "model_output_raw": "N/A",
            "model_output_processed": "N/A",
            "model_answer": "N/A",
            "ground_truth": item["answer"],
            "is_correct": False,
            "problem_id": item.get("id", "N/A")
        }

    pil_image = item['decoded_image']

    try:
        buffered = io.BytesIO()
        if pil_image.mode == 'RGBA':
            pil_image.save(buffered, format="PNG")
        else:
            pil_image.convert('RGB').save(buffered, format="PNG")

        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_image_url = f"data:image/png;base64,{encoded_image_text}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image_url},
                    },
                    {"type": "text", "text": f"You are an expert in math. Solve the question and output your final answer within \\boxed{{}}.\nExample: What is 1+1? \\boxed{{2}}\nQuestion: {item['problem_for_model']}"},
                ],
            },
        ]

        chat_response = client_instance.chat.completions.create(
            model="Qwen2-VL-7B-Instruct",
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=7000
        )
        response_text_raw = chat_response.choices[0].message.content
        response_text_processed = re.sub(r"\s*(<|>|/)\s*", r"\1", response_text_raw)
        model_answer = extract_boxed_content(response_text_processed)

        if model_answer is not None:
            is_correct = grade_answer(model_answer, item["answer"])
        else:
            print(f"Warning: No \\boxed{{}} content found for problem ID {item.get('id', 'N/A')}. Raw response: {response_text_raw[:200]}...")

    except Exception as e:
        error_message = str(e)
        print(f"Error processing response for problem ID {item.get('id', 'N/A')}: {e}")

    return {
        "input_problem": item['problem_for_model'],
        "image_present": True,
        "model_output_raw": response_text_raw,
        "model_output_processed": response_text_processed,
        "model_answer": model_answer,
        "ground_truth": item["answer"],
        "is_correct": is_correct,
        "problem_id": item.get("id", "N/A"),
        "error": error_message
    }

# Use ThreadPoolExecutor for concurrent requests
with ThreadPoolExecutor(max_workers=concurrent_requests_batch_size) as executor:
    future_to_item = {executor.submit(process_item, item, client): item for item in test_data}

    for future in tqdm(as_completed(future_to_item), total=len(test_data), desc="Processing problems"):
        item_identity = future_to_item[future]
        try:
            result = future.result()
            results.append(result)

            if result.get("is_correct"):
                acc_dict["correct"] += 1
            if not result.get("image_data_missing", False):
                acc_dict["num"] += 1
        except Exception as exc:
            problem_str_for_error_report = item_identity["question"]
            if len(item_identity["options"]) > 0:
                options_string_part_err_report = "Directly answer with letter in Options:\n" + "\n".join(item_identity["options"])
                problem_str_for_error_report = item_identity["question"] + "\n" + options_string_part_err_report

            print(f"Item {item_identity.get('id', 'N/A')} generated an exception during future.result(): {exc}")
            results.append({
                "input_problem": problem_str_for_error_report,
                "image_present": bool(item_identity.get('decoded_image')),
                "error": str(exc),
                "model_output_raw": "N/A (Exception in future.result())",
                "model_output_processed": "N/A (Exception in future.result())",
                "model_answer": "N/A",
                "ground_truth": item_identity["answer"],
                "is_correct": False,
                "problem_id": item_identity.get("id", "N/A")
            })

output_data = {
    "accuracy_stats": acc_dict,
    "results": results
}

print("--- Accuracy Stats ---")
print(json.dumps(acc_dict, indent=2))
print("----------------------")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

accuracy = acc_dict["correct"] / acc_dict["num"] * 100 if acc_dict["num"] > 0 else 0
print(f"Accuracy: {accuracy:.2f}% ({acc_dict['correct']}/{acc_dict['num']})")
print(f"Results saved to {output_file}")
