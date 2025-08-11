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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        image_dir: Optional[str] = None,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.image_dir = image_dir
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts, desc="Filtering overlong prompts", num_proc=16
            )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key] or []
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            resized_images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels) for image in images
            ] or None
            model_inputs = self.processor(resized_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        获取并处理单个数据样本，并行生成多模态和纯文本两组数据。
        """
        example = self.dataset[index]
        result = {}

        # --- 1. 处理多模态部分 (图像 + 问题) ---
        if self.image_key in example and self.problem_key_img in example:
            mm_messages = [{"role": "user", "content": example[self.problem_key_img]}]
            mm_prompt = self.processor.apply_chat_template(mm_messages, add_generation_prompt=True, tokenize=False)
            
            images = example[self.image_key]
            if self.image_dir is not None and len(images) > 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]
            
            resized_images = [process_image(img, min_pixels=self.min_pixels, max_pixels=self.max_pixels) for img in images] or None
            
            mm_model_inputs = self.processor(resized_images, [mm_prompt], add_special_tokens=False, return_tensors="pt")
            mm_input_ids = mm_model_inputs.pop("input_ids")[0]
            mm_attention_mask = mm_model_inputs.pop("attention_mask")[0]
            
            if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                mm_position_ids = get_rope_index(self.processor, input_ids=mm_input_ids, image_grid_thw=mm_model_inputs.get("image_grid_thw"), attention_mask=mm_attention_mask)
            else:
                mm_position_ids = torch.clip(mm_attention_mask.cumsum(dim=0) - 1, min=0)
            
            mm_input_ids, mm_attention_mask, mm_position_ids = VF.postprocess_data(
                input_ids=mm_input_ids, attention_mask=mm_attention_mask, position_ids=mm_position_ids,
                max_length=self.max_mm_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation=self.truncation
            )
            
            result["mm_input_ids"] = mm_input_ids
            result["mm_attention_mask"] = mm_attention_mask
            result["mm_position_ids"] = mm_position_ids
            result["mm_ground_truth"] = example[self.answer_key_img]
            result["multi_modal_data"] = {"images": images} # 保留原始图像信息

        # --- 2. 处理纯文本部分 ---
        if self.problem_key_txt in example:
            text_messages = [{"role": "user", "content": example[self.problem_key_txt]}]
            text_prompt = self.tokenizer.apply_chat_template(text_messages, add_generation_prompt=True, tokenize=False)
            
            text_model_inputs = self.tokenizer([text_prompt], add_special_tokens=False, return_tensors="pt")
            text_input_ids = text_model_inputs.pop("input_ids")[0]
            text_attention_mask = text_model_inputs.pop("attention_mask")[0]
            
            text_position_ids = torch.clip(text_attention_mask.cumsum(dim=0) - 1, min=0)
            
            text_input_ids, text_attention_mask, text_position_ids = VF.postprocess_data(
                input_ids=text_input_ids, attention_mask=text_attention_mask, position_ids=text_position_ids,
                max_length=self.max_text_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation=self.truncation
            )

            result["text_input_ids"] = text_input_ids
            result["text_attention_mask"] = text_attention_mask
            result["text_position_ids"] = text_position_ids
            result["text_ground_truth"] = example[self.answer_key_txt]

        return result