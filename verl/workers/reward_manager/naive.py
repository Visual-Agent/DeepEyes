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

from collections import defaultdict, deque

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import json
import datetime

class NaiveRewardManager:
    """The reward manager with historical tool call statistics support."""

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source",
        history_window_size=10,
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.history_window_size = history_window_size
        
        # 历史统计数据：使用 deque 实现滑动窗口
        # 每个元素是一个字典，包含：
        # - avg_tool_calls: 平均工具调用次数
        # - tool_success_rate: 工具调用成功率（使用工具且答案正确的比例）
        # - tool_usage_rate: 工具使用率（使用工具的样本比例）
        self.tool_call_history = deque(maxlen=history_window_size)
        
        self.step_cnt = 0
        self.current_batch_stats = None  # 当前批次的统计数据

    def _count_tool_calls(self, response_str: str) -> int:
        """
        统计响应中的工具调用次数
        这里使用与 vl_agent.py 中相同的统计方式
        """
        # 统计视觉工具调用次数
        vision_tool_count = response_str.count("<|vision_start|><|image_pad|>")
        
        # 也可以统计其他工具调用，如 <tool_call> 标签
        tool_call_count = response_str.count("<tool_call>")
        
        # 返回总工具调用次数
        return vision_tool_count + tool_call_count
    
    def _get_history_stats(self) -> dict:
        """
        获取历史统计数据
        返回滑动窗口内的平均统计值
        """
        if not self.tool_call_history:
            # 如果没有历史数据，返回默认值
            return {
                "avg_tool_calls": 1.0,  # 默认期望的工具调用次数
                "tool_success_rate": 0.5,  # 默认成功率
                "tool_usage_rate": 0.8,  # 默认使用率
                "history_size": 0
            }
        
        # 计算滑动窗口内的平均值
        total_avg_tool_calls = sum(stat["avg_tool_calls"] for stat in self.tool_call_history)
        total_tool_success_rate = sum(stat["tool_success_rate"] for stat in self.tool_call_history)
        total_tool_usage_rate = sum(stat["tool_usage_rate"] for stat in self.tool_call_history)
        
        history_size = len(self.tool_call_history)
        
        return {
            "avg_tool_calls": total_avg_tool_calls / history_size,
            "tool_success_rate": total_tool_success_rate / history_size,
            "tool_usage_rate": total_tool_usage_rate / history_size,
            "history_size": history_size
        }
    
    def _update_history(self, batch_stats: dict):
        """
        更新历史统计数据
        """
        self.tool_call_history.append(batch_stats)
        self.current_batch_stats = batch_stats

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        action_or_attn_mask = data.batch['action_mask'] if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
        if 'env_reward' in data.batch.keys():
            reward_tensor += data.batch['env_reward']
            print(f' [DEBUG reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}')

        # 获取历史统计数据
        history_stats = self._get_history_stats()
        
        # 当前批次的统计数据
        batch_tool_calls = []
        batch_tool_usage = []
        batch_tool_success = []
        
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            
            # 统计当前样本的工具调用次数
            tool_call_count = self._count_tool_calls(response_str)
            batch_tool_calls.append(tool_call_count)
            
            # 标记是否使用了工具
            used_tool = 1 if tool_call_count > 0 else 0
            batch_tool_usage.append(used_tool)

            # 调用奖励函数时传入历史统计数据
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                history_stats=history_stats,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # 记录工具使用是否成功（如果有acc信息）
                if "acc" in score:
                    # 如果使用了工具且答案正确，视为工具使用成功
                    tool_success = 1 if (used_tool and score["acc"] > 0.5) else (0 if used_tool else -1)
                    if tool_success >= 0:
                        batch_tool_success.append(tool_success)
                
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
                # 简单判断：如果奖励高且使用了工具，视为成功
                if used_tool:
                    tool_success = 1 if reward > 0.5 else 0
                    batch_tool_success.append(tool_success)

            reward_tensor[i, valid_response_length - 1] += reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print(f"[tool_calls]", tool_call_count)
                print(f"[history_stats]", history_stats)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

            self.step_cnt += 1
        
        # 计算当前批次的统计数据
        if len(data) > 0:
            avg_tool_calls = sum(batch_tool_calls) / len(data)
            tool_usage_rate = sum(batch_tool_usage) / len(data)
            tool_success_rate = sum(batch_tool_success) / len(batch_tool_success) if batch_tool_success else 0.0
            
            current_batch_stats = {
                "avg_tool_calls": avg_tool_calls,
                "tool_usage_rate": tool_usage_rate,
                "tool_success_rate": tool_success_rate,
                "batch_size": len(data)
            }
            
            # 更新历史统计
            self._update_history(current_batch_stats)
            
            # 将统计信息添加到奖励额外信息中
            reward_extra_info["batch_avg_tool_calls"] = [avg_tool_calls]
            reward_extra_info["batch_tool_usage_rate"] = [tool_usage_rate]
            reward_extra_info["batch_tool_success_rate"] = [tool_success_rate]
            reward_extra_info["history_avg_tool_calls"] = [history_stats["avg_tool_calls"]]
            reward_extra_info["history_tool_success_rate"] = [history_stats["tool_success_rate"]]
            
            print(f" [Batch Stats] avg_tool_calls={avg_tool_calls:.2f}, "
                  f"tool_usage_rate={tool_usage_rate:.2f}, "
                  f"tool_success_rate={tool_success_rate:.2f}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
