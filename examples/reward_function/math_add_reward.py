# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List, Literal

from mathruler.grader import extract_boxed_content, grade_answer

# --- Constants for Refusal Messages (for clarity and easy modification) ---
# Main target phrases for matching
REFUSAL_MISSING_PHRASE = "refused to answer, because the image is missing"
REFUSAL_IRRELEVANT_PHRASE = "refused to answer, because the image is redundant and irrelevant"


def is_correct_refusal(
    response: str,
    refusal_type: Literal["missing", "irrelevant"],
    allow_flexible_matching: bool = True
) -> bool:
    """
    Checks if the response is a correct and direct refusal.
    This function is designed to be robust against minor LLM output variations.
    """
    # Normalize the response for robust comparison: lowercase, strip whitespace, remove trailing period.
    normalized_response = response.strip().lower().rstrip(".")

    if refusal_type == "missing":
        target_phrase = REFUSAL_MISSING_PHRASE
    elif refusal_type == "irrelevant":
        target_phrase = REFUSAL_IRRELEVANT_PHRASE
    else:
        return False

    # --- Matching Patterns (from strictest to most flexible) ---

    # Pattern 1: Exact match after normalization (the ideal case)
    if normalized_response == target_phrase:
        return True

    if allow_flexible_matching:
        # Pattern 2: Check if the response contains the core phrase and is short.
        # This allows for minor variations but punishes long, rambling refusals.
        
        # if target_phrase in normalized_response and len(normalized_response) < len(target_phrase) + 15: # 这里是问题导致think消失
        #     return True
        
        # Pattern 3: Use regex for more flexibility (e.g., tolerating different punctuation or words)
        # This pattern checks for the key concepts: "refused", "answer", and the reason.
        if refusal_type == "missing":
            # Matches "refused...answer...image...missing"
            pattern = re.compile(r"refused.*answer.*image.*missing", re.IGNORECASE | re.DOTALL)
        else: # irrelevant
            # Matches "refused...answer...image.*(irrelevant|redundant)"
            pattern = re.compile(r"refused.*answer.*image.*(irrelevant|redundant)", re.IGNORECASE | re.DOTALL)
        
        if re.search(pattern, response):# and len(response) < 100: # Ensure it's still a concise refusal
            return True

    return False


def has_reasoning_leakage(response: str) -> bool:
    """Checks if the response contains forbidden reasoning or formatting."""
    return "<think>" in response or "</think>" in response or "\\boxed" in response


def format_reward(response: str) -> float:
    """Checks for the standard <think>...</think>\\boxed{...} format."""
    pattern = re.compile(r"<think>.*</think>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def format_reward_answer(response: str) -> float:
    """Checks for the standard <think>...</think>\\boxed{...} format."""
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Calculates accuracy for standard solvable problems."""
    answer = extract_boxed_content(response)
    # Penalize if the model refuses a solvable problem
    if "refused to answer" in response.lower():
        return 0.0
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Computes the reward score based on the new prompt logic.
    - `rollout_type='without_image'`: Expects a "missing image" refusal.
    - `rollout_type='irrelevant_image'`: Expects an "irrelevant image" refusal.
    - No `rollout_type` or other types: Expects a standard solution.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        rollout_type = reward_input.get("rollout_type")

        format_score = 0.0
        accuracy_score = 0.0

        # --- Case 1: The problem REQUIRES an image, but it's missing ---
        if rollout_type == "without_image":
            # Perfect score if it's a correct, direct refusal with no leakage.
            if is_correct_refusal(response, "missing"): # and not has_reasoning_leakage(response):
                accuracy_score = 1.0
                format_score = format_reward(response)
            # Otherwise, the score remains 0 (punishing wrong attempts or leaky refusals).

        # --- Case 2: The problem has an IRRELEVANT image ---
        # NOTE: You need to label your dataset with this new `rollout_type`.
        elif rollout_type == "text_with_image":
            # Perfect score if it's a correct, direct refusal with no leakage.
            if is_correct_refusal(response, "irrelevant"): # and not has_reasoning_leakage(response):
                accuracy_score = 1.0
                format_score = format_reward(response)
            # Otherwise, score is 0.

        # --- Case 3: The problem is standard and solvable ---
        else:
            accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
            format_score = format_reward_answer(response)

        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
