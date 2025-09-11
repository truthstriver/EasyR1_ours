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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, KLController, compute_kl, get_kl_controller
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics


import torch
def modify_raw_prompt_ids_v2(raw_prompt_ids, target_pair=(872, 198), insert_sequence=(151652, 151655, 151653)):
    """
    查找一个序列中连续的 ID 对，并在其后插入指定的 token 序列。

    Args:
        raw_prompt_ids (list or np.array): 包含一个或多个 token ID 序列的列表。
        target_pair (tuple, optional): 需要查找的连续 ID 对。默认为 (872, 198)。
        insert_sequence (tuple, optional): 需要插入的 token ID 序列。默认为 (151652, 151655, 151653)。

    Returns:
        np.array: 修改后的序列数组。由于序列长度可能发生变化，dtype 设置为 object。
    """
    modified_sequences = []

    for seq in raw_prompt_ids:
        seq_list = list(seq)
        
        # 从头开始查找目标 ID 对的第一个出现位置
        found_idx = -1
        for i in range(len(seq_list) - 1):
            if seq_list[i] == target_pair[0] and seq_list[i+1] == target_pair[1]:
                found_idx = i
                break  # 找到后即停止搜索

        # 如果找到了目标 ID 对
        if found_idx != -1:
            # 确定插入点的位置（在目标 ID 对之后）
            insert_point = found_idx + 2
            # 使用切片赋值来插入新序列
            seq_list[insert_point:insert_point] = list(insert_sequence)
        
        modified_sequences.append(seq_list)

    # 使用 dtype=object 的 NumPy 数组来处理可能变化的序列长度
    return np.array(modified_sequences, dtype=object)
def modify_raw_prompt_ids(raw_prompt_ids, start_token=151652, end_token=151653, insert_token=151643, pad_token=0):
    modified_sequences = []

    for seq in raw_prompt_ids:
        seq_list = list(seq)  # 确保是 Python list

        # Step 1: 找到 start_token 和 end_token 的位置
        try:
            start_idx = seq_list.index(start_token)
            end_idx = seq_list.index(end_token, start_idx)
        except ValueError:
            # 如果找不到 start_token 或 end_token，保留原序列
            modified_sequences.append(seq_list)
            continue

        # Step 2: 删除 [start_token, end_token] 区间内的所有 token（包含两端）
        deleted_tokens = seq_list[start_idx : end_idx + 1]
        del seq_list[start_idx : end_idx + 1]
        num_deleted = len(deleted_tokens)

        # Step 3: 在开头插入相同数量的 insert_token
        seq_list = [insert_token] * num_deleted + seq_list

        # Step 4: 截断或填充以保持原始长度不变
        original_length = len(seq)
        if len(seq_list) > original_length:
            seq_list = seq_list[:original_length]
        elif len(seq_list) < original_length:
            seq_list += [pad_token] * (original_length - len(seq_list))

        modified_sequences.append(seq_list)

    return np.array(modified_sequences, dtype=object)
    


def modify_input_ids(input_ids, start_token=151652, end_token=151653, insert_token=151643):
    modified_input_ids = []
    for seq in input_ids:
        seq_list = seq.tolist()  # 转换为 Python list 方便处理
        
        # 1. 找到 start_token 和 end_token 的位置
        try:
            start_idx = seq_list.index(start_token)
            end_idx = seq_list.index(end_token, start_idx)
        except ValueError:  # 如果没有找到 start_token 或 end_token，则跳过
            modified_input_ids.append(seq)
            continue
        
        # 2. 删除 start_token 到 end_token（含）之间的所有 token
        del seq_list[start_idx : end_idx + 1]
        num_deleted = end_idx - start_idx + 1
        
        # 3. 在开头插入相同数量的 insert_token
        seq_list = [insert_token] * num_deleted + seq_list
        
        # 4. 保持长度不变（可能需要截断或填充）
        if len(seq_list) > len(seq):  # 如果太长，截断
            seq_list = seq_list[:len(seq)]
        elif len(seq_list) < len(seq):  # 如果太短，用 pad_token 填充（假设 pad_token=0）
            seq_list += [0] * (len(seq) - len(seq_list))
        
        modified_input_ids.append(torch.tensor(seq_list, dtype=seq.dtype, device=seq.device))
    
    return torch.stack(modified_input_ids)
def modify_input_ids_v2(input_ids, pad_token_id=0):
    """
    在 input_ids 中找到连续的 [872, 198] token，然后在其后插入 [151652, 151655, 151653]，
    并从序列开头删除三个填充符以保持长度不变。

    Args:
        input_ids (torch.Tensor): 输入的 token ID 张量，形状为 (batch_size, sequence_length)。
        pad_token_id (int, optional): 用于填充的 token ID。默认为 0。

    Returns:
        torch.Tensor: 修改后的 token ID 张量。
    """
    modified_input_ids = []
    # 要查找的序列
    sequence_to_find = [872, 198]
    # 要插入的序列
    sequence_to_insert = [151652, 151655, 151653]
    num_to_insert = len(sequence_to_insert)

    for seq in input_ids:
        # 将 tensor 转换为 Python list 以便进行灵活处理
        seq_list = seq.tolist()
        
        # 寻找 sequence_to_find 的位置
        found_idx = -1
        for i in range(len(seq_list) - len(sequence_to_find) + 1):
            if seq_list[i:i+len(sequence_to_find)] == sequence_to_find:
                found_idx = i
                break
        
        # 如果找到了目标序列
        if found_idx != -1:
            # 1. 在找到的序列后面插入新的 token
            # 计算插入点的位置，即 [872, 198] 的后面
            insert_position = found_idx + len(sequence_to_find)
            # 执行插入操作
            seq_list = seq_list[:insert_position] + sequence_to_insert + seq_list[insert_position:]

            # 2. 从开头删除溢出的 token（假设是 padding tokens）
            # 为了保持序列长度不变，需要删除与插入数量相同的 token
            # 这里我们假设被删除的是序列开头的 padding tokens
            final_seq_list = seq_list[num_to_insert:]

            # 3.（可选）验证并确保长度严格不变
            if len(final_seq_list) != len(seq):
                # 如果插入操作导致原序列末尾的 token 被挤出，上面的切片操作已经处理了
                # 如果因为其他原因长度不匹配，这里可以添加警告或错误处理
                # 在这个逻辑下，长度应该始终保持不变
                pass
            
            # 将修改后的 list 转换回 tensor
            modified_seq = torch.tensor(final_seq_list, dtype=seq.dtype, device=seq.device)
            modified_input_ids.append(modified_seq)
        else:
            # 如果没有找到目标序列，则保持原样
            modified_input_ids.append(seq)
            
    # 将所有处理过的序列重新堆叠成一个 tensor
    return torch.stack(modified_input_ids)
def concat_data_protos(obj1: DataProto, obj2: DataProto) -> DataProto:
    """
    Concatenates two DataProto objects.

    Args:
        obj1: The first DataProto object.
        obj2: The second DataProto object.

    Returns:
        A new DataProto object containing the combined data.
    """
    if not isinstance(obj1, DataProto) or not isinstance(obj2, DataProto):
        raise TypeError("Both inputs must be DataProto objects.")

    # 1. Concatenate the TensorDict `batch` using torch.cat
    concatenated_batch = torch.cat([obj1.batch, obj2.batch], dim=0)

    # 2. Concatenate the `non_tensor_batch` dictionary manually
    new_non_tensor_batch = {}
    for key in obj1.non_tensor_batch:
        val1 = obj1.non_tensor_batch[key]
        if key=="multi_modal_data":
            val2 = np.array([{"images":[]}] * len(obj2.batch))
        else:
            val2 = obj2.non_tensor_batch[key]
        
        if isinstance(val1, np.ndarray):
            new_non_tensor_batch[key] = np.concatenate([val1, val2])
        else:
            raise TypeError(f"Unhandled type for key '{key}': {type(val1)}")

    # 3. Handle meta_info concatenation
    new_meta_info = {}
    for key in obj1.meta_info:
        val1 = obj1.meta_info[key]
        val2 = obj2.meta_info[key]
        
        if key == 'global_token_num':
            # Special handling for token_num: concatenate the lists
            new_meta_info[key] = val1 + val2
        elif isinstance(val1, list):
            # For regular lists, concatenate them
            new_meta_info[key] = val1 + val2
        elif isinstance(val1, (int, float, str)):
            # For scalar values, verify they're identical and keep one
            if val1 != val2:
                raise ValueError(f"meta_info values for key '{key}' differ between objects")
            new_meta_info[key] = val1
        elif isinstance(val1, dict):
            # For dictionaries, recursively merge them
            if val1.keys() != val2.keys():
                raise ValueError(f"Dictionary keys in meta_info '{key}' differ between objects")
            merged_dict = {}
            for sub_key in val1:
                if val1[sub_key] != val2[sub_key]:
                    raise ValueError(f"Dictionary values in meta_info '{key}.{sub_key}' differ between objects")
                merged_dict[sub_key] = val1[sub_key]
            new_meta_info[key] = merged_dict
        else:
            raise TypeError(f"Unhandled type in meta_info for key '{key}': {type(val1)}")

    # Create and return the new, combined DataProto object
    return DataProto(
        batch=concatenated_batch,
        non_tensor_batch=new_non_tensor_batch,
        meta_info=new_meta_info
    )



def split_data_proto(obj: DataProto):
    """
    将一个 DataProto 对象按奇偶索引拆分为两个。

    Args:
        obj: 要拆分的 DataProto 对象。

    Returns:
        一个包含两个新的 DataProto 对象的元组，
        第一个包含所有偶数索引的数据，第二个包含所有奇数索引的数据。
    """
    if not isinstance(obj, DataProto):
        raise TypeError("输入必须是 DataProto 对象。")

    total_size = len(obj.batch)
    if total_size == 0:
        raise ValueError("无法拆分空的 DataProto 对象。")

    # 1. 按奇偶索引拆分 TensorDict `batch`
    batch1 = obj.batch[::2]
    batch2 = obj.batch[1::2]

    # 2. 按奇偶索引拆分 `non_tensor_batch` 中的Numpy数组
    new_non_tensor_batch1 = {}
    new_non_tensor_batch2 = {}
    for key, val in obj.non_tensor_batch.items():
        if isinstance(val, np.ndarray):
            new_non_tensor_batch1[key] = val[::2]
            new_non_tensor_batch2[key] = val[1::2]
        else:
            raise TypeError(f"在 non_tensor_batch 中遇到非预期的类型 '{key}': {type(val)}")

    # 3. 处理 meta_info 的拆分
    new_meta_info1 = {}
    new_meta_info2 = {}
    for key, val in obj.meta_info.items():
        if isinstance(val, list):
            # 对于列表，按奇偶拆分
            new_meta_info1[key] = val[::2]
            new_meta_info2[key] = val[1::2]
        elif isinstance(val, (int, float, str, dict)):
            # 对于标量和字典，在合并时它们必须是相同的，所以直接复制
            new_meta_info1[key] = val
            new_meta_info2[key] = val
        else:
            raise TypeError(f"在 meta_info 中遇到非预期的类型 '{key}': {type(val)}")

    # 创建并返回两个新的 DataProto 对象
    obj1 = DataProto(
        batch=batch1,
        non_tensor_batch=new_non_tensor_batch1,
        meta_info=new_meta_info1
    )
    
    obj2 = DataProto(
        batch=batch2,
        non_tensor_batch=new_non_tensor_batch2,
        meta_info=new_meta_info2
    )

    return obj1, obj2

def split_data_proto_by_ratio(obj: DataProto, n: float):
    """
    将一个 DataProto 对象按比例 n 进行划分。

    如果 n 为正 (0 < n <= 1)，则返回包含前 n 比例数据的 DataProto 对象。
    如果 n 为负 (-1 <= n < 0)，则返回包含后 |n| 比例数据的 DataProto 对象。

    Args:
        obj: 要划分的 DataProto 对象。
        n: 划分比例，取值范围为 [-1, 1] 且不为 0。

    Returns:
        一个包含划分后数据的新 DataProto 对象。
    """
    if not isinstance(obj, DataProto):
        raise TypeError("输入必须是 DataProto 对象。")
    if not isinstance(n, float) or not (-1 <= n <= 1) or n == 0:
        raise ValueError("参数 'n' 必须是 [-1, 1] 范围内的浮点数且不为 0。")

    total_size = len(obj.batch)
    if total_size == 0:
        raise ValueError("无法划分空的 DataProto 对象。")

    if n > 0:
        # 正比例：取列表前 n * total_size 的部分
        # 例如 n=0.8, total_size=10, end_idx=8, 切片为 [0:8]
        end_idx = int(total_size * n)
        slicer = slice(None, end_idx)
    else: # n < 0
        # 负比例：取列表后 |n| * total_size 的部分
        # 例如 n=-0.2, total_size=10, start_idx=8, 切片为 [8:]
        start_idx = int(total_size * (1 + n))
        slicer = slice(start_idx, None)
        
    # 1. 按比例切片 TensorDict `batch`
    new_batch = obj.batch[slicer]

    # 2. 按比例切片 `non_tensor_batch` 中的Numpy数组
    new_non_tensor_batch = {}
    for key, val in obj.non_tensor_batch.items():
        if isinstance(val, np.ndarray):
            new_non_tensor_batch[key] = val[slicer]
        else:
            raise TypeError(f"在 non_tensor_batch 中遇到非预期的类型 '{key}': {type(val)}")

    # 3. 按比例处理 meta_info
    new_meta_info = {}
    for key, val in obj.meta_info.items():
        if isinstance(val, list):
            # 对于列表，按比例切片
            new_meta_info[key] = val[slicer]
        elif isinstance(val, (int, float, str, dict)):
            # 对于标量和字典，它们不与 batch 的长度关联，直接复制
            new_meta_info[key] = val
        else:
            raise TypeError(f"在 meta_info 中遇到非预期的类型 '{key}': {type(val)}")

    # 创建并返回新的 DataProto 对象
    new_obj = DataProto(
        batch=new_batch,
        non_tensor_batch=new_non_tensor_batch,
        meta_info=new_meta_info
    )

    return new_obj

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data

class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    # def _validate(self) -> Dict[str, Any]:
    #     reward_tensor_lst = []
    #     # Lists to collect samples for the table
    #     sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
    #     reward_metrics_lst = defaultdict(list)
    #     print("Start validation...")
    #     self.actor_rollout_ref_wg.prepare_rollout_engine()
    #     for batch_dict in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(batch_dict)
    #         test_gen_batch = test_batch.pop(
    #             batch_keys=["input_ids", "attention_mask", "position_ids"],
    #             non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
    #         )
    #         repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
    #         test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
    #         test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
    #         test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels

    #         test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
    #         test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
    #         test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

    #         # repeat to align with repeated responses in rollout
    #         test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
    #         test_batch = test_batch.union(test_output_gen_batch)

    #         # evaluate using reward_function
    #         reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

    #         # store generations
    #         input_ids = test_batch.batch["prompts"]
    #         input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    #         output_ids = test_batch.batch["responses"]
    #         output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    #         scores = reward_tensor.sum(-1).cpu().tolist()
    #         sample_inputs.extend(input_texts)
    #         sample_outputs.extend(output_texts)
    #         sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
    #         sample_scores.extend(scores)

    #         reward_tensor_lst.append(reward_tensor)
    #         for key, value in reward_metrics.items():
    #             reward_metrics_lst[key].extend(value)

    #     self.actor_rollout_ref_wg.release_rollout_engine()
    #     self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
    #     self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
    #     val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
    #     print("Finish validation.")
    #     return {"val/reward_score": self.val_reward_score, **val_reward_metrics}
    def _validate(self) -> Dict[str, Any]:
        """
        在验证集上评估模型，并实现四种不同的数据处理方式：
        1. with_image: 多模态输入，包含图像。
        2. without_image: 多模态输入，但移除图像。
        3. text_with_image: 纯文本输入，但添加 <image> 标记。
        4. text_without_image: 纯文本输入，不含 <image> 标记。
        """
        # 初始化用于收集所有四种数据类型指标的容器
        all_reward_metrics_lst = defaultdict(lambda: defaultdict(list))
        all_reward_tensor_lst = defaultdict(list)
        all_samples = defaultdict(lambda: {"inputs": [], "outputs": [], "labels": [], "scores": []})

        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()

        for batch_dict in self.val_dataloader:
            # 1. 准备初始数据批次
            base_dataproto = DataProto.from_single_dict(batch_dict)
            
            # 将数据分为多模态部分和纯文本部分
            batch_with_img_base, batch_pure_text_base = split_data_proto(base_dataproto)

            # 2. 创建四种数据变体用于生成
            # a) 带有图像的数据 (原始多模态)
            gen_batch_with_img = batch_with_img_base.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )

            # b) 移除图像的数据
            gen_batch_without_img = deepcopy(gen_batch_with_img)
            gen_batch_with_img = split_data_proto_by_ratio(gen_batch_with_img,1-self.config.worker.rollout.split_ratio)
            gen_batch_without_img = split_data_proto_by_ratio(gen_batch_without_img,-self.config.worker.rollout.split_ratio)
            _ = gen_batch_without_img.pop(batch_keys=[], non_tensor_batch_keys=["multi_modal_data"])
            gen_batch_without_img.batch["input_ids"] = modify_input_ids(gen_batch_without_img.batch["input_ids"])
            gen_batch_without_img.non_tensor_batch["raw_prompt_ids"] = modify_raw_prompt_ids(
                gen_batch_without_img.non_tensor_batch["raw_prompt_ids"]
            )

            # c) 纯文本数据 (原始纯文本)
            text_gen_batch_without_img = batch_pure_text_base.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )


            # d) 添加了<image>标记的纯文本数据
            text_gen_batch_with_img = deepcopy(text_gen_batch_without_img)
            text_gen_batch_with_img = split_data_proto_by_ratio(text_gen_batch_with_img,-self.config.worker.rollout.split_ratio)
            text_gen_batch_without_img = split_data_proto_by_ratio(text_gen_batch_without_img,1-self.config.worker.rollout.split_ratio)
            _ = text_gen_batch_without_img.pop(batch_keys=[],non_tensor_batch_keys=["multi_modal_data"],meta_info_keys=[])
            

            text_gen_batch_without_img.batch["input_ids"] = modify_input_ids(text_gen_batch_without_img.batch["input_ids"])
            text_gen_batch_without_img.non_tensor_batch["raw_prompt_ids"] = modify_raw_prompt_ids(
                text_gen_batch_with_img.non_tensor_batch["raw_prompt_ids"]
            )
            
            # 将四种变体及其对应的原始数据批次分组
            validation_cases = {
                "with_image": (gen_batch_with_img, batch_with_img_base),
                "without_image": (gen_batch_without_img, batch_with_img_base),
                "text_with_image": (text_gen_batch_with_img, batch_pure_text_base),
                "text_without_image": (text_gen_batch_without_img, batch_pure_text_base),
            }

            # 3. 对每种数据变体进行处理和评估
            for case_name, (gen_batch, original_batch) in validation_cases.items():
                gen_batch.meta_info = self.config.worker.rollout.val_override_config
                gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
                gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
                
                batch_size = len(gen_batch.batch["input_ids"])
                gen_batch.meta_info["rollout_type"] = [case_name] * batch_size
                
                # 填充、生成序列、并移除填充
                gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
                test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

                # 合并生成结果与原始数据
                final_batch = original_batch.union(test_output_gen_batch)

                # 计算奖励
                reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(final_batch))

                # 解码并存储样本
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in final_batch.batch["prompts"]]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in final_batch.batch["responses"]]
                scores = reward_tensor.sum(-1).cpu().tolist()
                
                all_samples[case_name]["inputs"].extend(input_texts)
                all_samples[case_name]["outputs"].extend(output_texts)
                all_samples[case_name]["labels"].extend(final_batch.non_tensor_batch["ground_truth"].tolist())
                all_samples[case_name]["scores"].extend(scores)

                # 收集奖励和指标
                all_reward_tensor_lst[case_name].append(reward_tensor)
                for key, value in reward_metrics.items():
                    all_reward_metrics_lst[case_name][key].extend(value)

        self.actor_rollout_ref_wg.release_rollout_engine()

        # 4. 聚合和记录结果
        final_val_metrics = {}
        total_scores = []
        
        for case_name, samples in all_samples.items():
            # 记录每种情况下的生成样本
            self._maybe_log_val_generations(
                samples["inputs"], samples["outputs"], samples["labels"], samples["scores"], log_prefix=f"val_{case_name}"
            )

            # 计算并记录每种情况的奖励分数
            reward_score = torch.cat(all_reward_tensor_lst[case_name], dim=0).sum(-1).mean().item()
            final_val_metrics[f"val/{case_name}/reward_score"] = reward_score
            total_scores.extend(samples["scores"])

            # 聚合和记录其他奖励指标
            reward_metrics = reduce_metrics(all_reward_metrics_lst[case_name])
            for key, value in reward_metrics.items():
                final_val_metrics[f"val/{case_name}/{key}_reward"] = value
        
        # 计算并记录一个总的平均奖励分数，以保持向后兼容性
        self.val_reward_score = np.mean(total_scores) if total_scores else 0.0
        final_val_metrics["val/reward_score"] = self.val_reward_score

        print("Finish validation.")
        return final_val_metrics

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {"min_pixels": self.config.data.min_pixels, "max_pixels": self.config.data.max_pixels}
            new_batch_tmp: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        
            # data are interleaved, 1 with img , and 1 without img
            # pop those keys for generation
            new_batch, batch_pure_text =  split_data_proto(new_batch_tmp)

            # INPUT 1
            gen_batch_with_img = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels"],
            )
            # INPUT 2
            gen_batch_without_img = deepcopy(gen_batch_with_img)  # avoid modifying the original batch
            # 取 1-n的比例作为正例
            gen_batch_with_img = split_data_proto_by_ratio(gen_batch_with_img,1-self.config.worker.rollout.split_ratio)
            # 取后n的比例作为负例
            gen_batch_without_img = split_data_proto_by_ratio(gen_batch_without_img,-self.config.worker.rollout.split_ratio)
            _ = gen_batch_without_img.pop(batch_keys=[],non_tensor_batch_keys=["multi_modal_data"],meta_info_keys=[])
            
            # INPUT 3
            text_gen_batch_with_img = batch_pure_text.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels"],
            )
            # INPUT 4
            text_gen_batch_without_img = deepcopy(text_gen_batch_with_img)  # avoid modifying the original batch

            text_gen_batch_with_img = split_data_proto_by_ratio(text_gen_batch_with_img,-self.config.worker.rollout.split_ratio)
            text_gen_batch_without_img = split_data_proto_by_ratio(text_gen_batch_without_img,1-self.config.worker.rollout.split_ratio)

            _ = text_gen_batch_without_img.pop(batch_keys=[],non_tensor_batch_keys=["multi_modal_data"],meta_info_keys=[])
            
            # 应用修改 去掉图<image token>也得去掉
            gen_batch_without_img.batch["input_ids"] = modify_input_ids(gen_batch_without_img.batch["input_ids"])
            gen_batch_without_img.non_tensor_batch["raw_prompt_ids"] = modify_raw_prompt_ids(
                gen_batch_without_img.non_tensor_batch["raw_prompt_ids"])
            
            # 文本加图要有<image token>
            text_gen_batch_without_img.batch["input_ids"] = modify_input_ids(text_gen_batch_without_img.batch["input_ids"])
            text_gen_batch_without_img.non_tensor_batch["raw_prompt_ids"] = modify_raw_prompt_ids(
                text_gen_batch_without_img.non_tensor_batch["raw_prompt_ids"])


            #　比例参数计算
            # n_with_img = int(self.config.worker.rollout.n * (1-self.config.worker.rollout.split_ratio))

            # #　盈利参数计算
            # gen_batch_with_img.meta_info["n"] = n_with_img
            # gen_batch_without_img.meta_info["n"] = self.config.worker.rollout.n - n_with_img

            # text_gen_batch_without_img.meta_info['n'] = n_with_img
            # text_gen_batch_with_img.meta_info["n"] = self.config.worker.rollout.n - n_with_img
            

            # generate a batch
            gen_batch_output_with_img = self.actor_rollout_ref_wg.generate_sequences(gen_batch_with_img)
            gen_batch_output_without_img = self.actor_rollout_ref_wg.generate_sequences(gen_batch_without_img)

            text_gen_batch_output_without_img = self.actor_rollout_ref_wg.generate_sequences(text_gen_batch_without_img) # origin
            text_gen_batch_output_with_img = self.actor_rollout_ref_wg.generate_sequences(text_gen_batch_with_img) # mod



            # if self.config.algorithm.adv_estimator == "remax":
            #     gen_baseline_batch = deepcopy(gen_batch)
            #     gen_baseline_batch.meta_info["temperature"] = 0
            #     gen_baseline_batch.meta_info["n"] = 1
            #     gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

            #     new_batch = new_batch.union(gen_baseline_output)
            #     reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            #     reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

            #     new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            #     new_batch.batch["reward_baselines"] = reward_baseline_tensor
            #     del gen_baseline_batch, gen_baseline_output
            
            # new_batch.non_tensor_batch["uid"] = np.array(
            #     [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            # )  # No more assign totally

            # re assign the uid to the batch
            new_batch_with_image = deepcopy(new_batch)
            new_batch_without_image = deepcopy(new_batch)

            text_new_batch_with_image = deepcopy(batch_pure_text)
            text_new_batch_without_image = deepcopy(batch_pure_text)

            new_batch_with_image.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch_with_image.batch))], dtype=object
            )
            new_batch_without_image.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch_with_image.batch),len(new_batch_with_image.batch) + len(new_batch_without_image.batch))],
                dtype=object,
            )

            text_new_batch_with_image.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(text_new_batch_with_image.batch))], dtype=object
            )
            text_new_batch_without_image.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(text_new_batch_without_image.batch))], dtype=object,
            )


            
            new_batch_with_image = new_batch_with_image.repeat(
                repeat_times=n_with_img, interleave=True
            )  # repeat to align with repeated responses in rollout
            
            new_batch_without_image = new_batch_without_image.repeat(
                repeat_times=self.config.worker.rollout.n - n_with_img, interleave=True
            )  # repeat to align with repeated responses in rollout
            
            # repeat pure text data
            text_new_batch_without_image = text_new_batch_without_image.repeat(
                repeat_times=n_with_img, interleave=True
            )  # repeat to align with repeated responses in rollout
            
            text_new_batch_with_image = text_new_batch_with_image.repeat(
                repeat_times=self.config.worker.rollout.n - n_with_img, interleave=True
            )  # repeat to align with repeated responses in rollout
            # new_batch = new_batch.union(gen_batch_output)
            new_batch_with_image = new_batch_with_image.union(gen_batch_output_with_img)                            # batch \times rollout.n
            new_batch_without_image = new_batch_without_image.union(gen_batch_output_without_img)                   # batch \times rollout.n
            text_new_batch_with_image = text_new_batch_with_image.union(text_gen_batch_output_with_img)             # batch \times rollout.n
            text_new_batch_without_image = text_new_batch_without_image.union(text_gen_batch_output_without_img)    # batch \times rollout.n
            # # filter group
            # if self.config.algorithm.online_filtering:
            #     reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            #     new_batch.batch["token_level_scores"] = reward_tensor
            #     for k, v in reward_metrics.items():
            #         all_metrics[k].extend(v)

            #     filter_scores = reward_metrics[self.config.algorithm.filter_key]
            #     uids = new_batch.non_tensor_batch["uid"]
            #     uid2scores = defaultdict(list)
            #     for uid, score in zip(uids, filter_scores):
            #         uid2scores[uid].append(score)

            #     uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
            #     kept_uids = [
            #         uid
            #         for uid, avg_score in uid2mean.items()
            #         if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
            #     ]
            #     kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
            #     new_batch = new_batch[kept_sample_idxs]

            # batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            # current_batch_size = len(batch) // self.config.worker.rollout.n
            # rollout_batch_size = self.config.data.rollout_batch_size


            # if current_batch_size < rollout_batch_size:
            #     print(f"{current_batch_size=} < {rollout_batch_size=}")
            #     max_try_make_batch = self.config.trainer.max_try_make_batch
            #     if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
            #         print(f"{num_try_make_batch=}. Continue generating...")
            #     else:
            #         raise ValueError(
            #             f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
            #         )
            # else:
            #     print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
            #     if self.config.algorithm.online_filtering:
            #         metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

            #     return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
            return new_batch_with_image, new_batch_without_image, text_new_batch_with_image, text_new_batch_without_image

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch_with_img, batch_without_img, text_new_batch_with_image, text_new_batch_without_image = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()
                    
                    
                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch_with_img, metrics=metrics)
                self._balance_batch(batch_without_img, metrics=metrics)
                self._balance_batch(text_new_batch_with_image, metrics=metrics, logging_prefix="text_global_seqlen")
                self._balance_batch(text_new_batch_without_image, metrics=metrics, logging_prefix="text_global_seqlen")


                # compute global valid tokens
                batch_with_img.meta_info["global_token_num"] = torch.sum(batch_with_img.batch["attention_mask"], dim=-1).tolist()
                batch_without_img.meta_info["global_token_num"] = torch.sum(batch_without_img.batch["attention_mask"], dim=-1).tolist()
                text_new_batch_with_image.meta_info["global_token_num"] = torch.sum(text_new_batch_with_image.batch["attention_mask"], dim=-1).tolist()
                text_new_batch_without_image.meta_info["global_token_num"] = torch.sum(text_new_batch_without_image.batch["attention_mask"], dim=-1).tolist()

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs_with_img = self.actor_rollout_ref_wg.compute_log_probs(batch_with_img)
                    batch_with_img = batch_with_img.union(old_log_probs_with_img)
                    
                    old_log_probs_without_img = self.actor_rollout_ref_wg.compute_log_probs(batch_without_img)
                    batch_without_img = batch_without_img.union(old_log_probs_without_img)

                    old_log_text_probs_with_img = self.actor_rollout_ref_wg.compute_log_probs(text_new_batch_with_image)
                    text_new_batch_with_image = text_new_batch_with_image.union(old_log_text_probs_with_img)

                    old_log_text_probs_without_img = self.actor_rollout_ref_wg.compute_log_probs(text_new_batch_without_image)
                    text_new_batch_without_image = text_new_batch_without_image.union(old_log_text_probs_without_img)




                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs_with_img = self.actor_rollout_ref_wg.compute_ref_log_probs(batch_with_img)
                        batch_with_img = batch_with_img.union(ref_log_probs_with_img)

                        ref_log_probs_without_img = self.actor_rollout_ref_wg.compute_ref_log_probs(batch_without_img)
                        batch_without_img = batch_without_img.union(ref_log_probs_without_img)

                        ref_log_text_probs_with_img = self.actor_rollout_ref_wg.compute_ref_log_probs(text_new_batch_with_image)
                        text_new_batch_with_image = text_new_batch_with_image.union(ref_log_text_probs_with_img)

                        ref_log_text_probs_without_img = self.actor_rollout_ref_wg.compute_ref_log_probs(text_new_batch_without_image)
                        text_new_batch_without_image = text_new_batch_without_image.union(ref_log_text_probs_without_img)




                batch = concat_data_protos(concat_data_protos(batch_with_img, batch_without_img), concat_data_protos(text_new_batch_with_image, text_new_batch_without_image))
                batch.meta_info["rollout_type"] = ["with_image"] * len(batch_with_img) + ["without_image"] * len(batch_without_img) + ["text_with_image"] * len(text_new_batch_with_image) + ["text_without_image"] * len(text_new_batch_without_image)

                
                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)
                        
                # # compute values
                # if self.use_critic:
                #     with timer("values", timing_raw):
                #         values = self.critic_wg.compute_values(batch)
                #         batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
