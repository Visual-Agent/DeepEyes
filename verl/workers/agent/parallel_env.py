import re
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.vision_utils import process_image, process_raw_image, process_video
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.agent.tool_envs import ToolBase

def _strip_system_block(text: str) -> str:
    """
    删除 text 中第一个 <|im_start|>system ... <|im_end|> 区块（含标签），
    并返回删除后的字符串。
    如果找不到匹配的开始或结束标签，则返回原文。
    """
    # 非贪婪匹配，匹配跨行
    pattern = r"<\|im_start\|>system.*?<\|im_end\|>"
    # 替换为空
    result = re.sub(pattern, "", text, flags=re.S)
    return result


def _concat_vllm_input(prompt_token_ids, response_token_ids, tokenizer=None):
    # NOTE: temporarily fix qwen-base oov issue
    if tokenizer is not None:
        max_token_id = max(tokenizer.get_vocab().values())
        tokenizer_size = len(tokenizer)
        max_token_id = max(max_token_id, tokenizer_size)
        valid_token_mask = torch.le(response_token_ids, max_token_id)
        response_token_ids = torch.masked_select(response_token_ids, valid_token_mask)

    if isinstance(prompt_token_ids, torch.Tensor):
        output_tensor = torch.cat([
            prompt_token_ids,
            response_token_ids.to(prompt_token_ids.device),
        ], dim=-1)
        return output_tensor.cpu().numpy().flatten().tolist()
    else:
        output_array = np.concatenate([
            prompt_token_ids,
            response_token_ids.cpu().numpy(),
        ], axis=-1)
        return output_array.flatten().tolist()


def _merge_multi_modal_inputs(mm_input, other):
    if not mm_input and not other:
        return {}
    elif len(mm_input) == 0 and len(other) > 0:
        return other
    elif len(mm_input) > 0 and len(other) == 0:
        return mm_input

    output_dict = {}
    for key in mm_input.keys():
        if key not in other.keys():
            output_dict[key] = mm_input[key]
            continue

        mm_value = mm_input[key]
        other_value = other.pop(key)
        if isinstance(mm_value, np.ndarray) and isinstance(other_value, np.ndarray):
            merged_value = np.concatenate([mm_value, other_value], axis=0)
        elif isinstance(mm_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            merged_value = torch.cat([mm_value, other_value], dim=0)
        else:
            raise ValueError(f"Invalid {type(mm_value)=}, {type(other_value)=}")

        output_dict[key] = merged_value
    return dict(**output_dict, **other)


def _preprocess_multi_modal_inputs(prompt_str, processor, **kwargs):
    if processor is None or "multi_modal_data" not in kwargs:
        return prompt_str, prompt_str, {}

    vllm_input_prompt = prompt_str.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    input_mm_data = kwargs.get("multi_modal_data", {"image": []})
    input_mm_data["image"] = [process_image(image) for image in input_mm_data['image']]
    model_inputs = processor(text=[vllm_input_prompt], images=input_mm_data["image"], return_tensors="pt")
    input_ids = model_inputs.pop("input_ids")[0]
    attention_mask = model_inputs.pop("attention_mask")[0]

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    mm_inputs = dict(model_inputs)
    return vllm_input_prompt, input_ids, mm_inputs


def agent_rollout_loop(config, vllm_engine, vllm_inputs, prompts, multi_modal_inputs, sampling_params):
    agent_sampling_params = sampling_params.clone()
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.spaces_between_special_tokens = False
    agent_sampling_params.n = 1
    agent_sampling_params.include_stop_str_in_output = True
    max_generated_tokens = min(config.agent.single_response_max_tokens, config.response_length)
    agent_sampling_params.max_tokens = max_generated_tokens

    # support custom stop specified in dataset, like </search>, ```, etc.
    custom_stop = list(config.agent.custom_stop)
    if custom_stop:
        prev_stop = sampling_params.stop if sampling_params.stop else []
        agent_sampling_params.stop = prev_stop + custom_stop
        print(f' [DEBUG stop] {type(prev_stop)=}, {type(custom_stop)=}, {type(agent_sampling_params.stop)=}')

    # Refer to: https://github.com/vllm-project/vllm/issues/1728
    # and https://github.com/vllm-project/vllm/issues/15976
    # def process_bad_tokens(token_ids, logits, exclude_token_ids=[]):
    #     for token_id in exclude_token_ids:
    #         logits[token_id] = -9999.999
    #     return logits

    # # NOTE: tmp for visual agent!
    # exclude_func = partial(process_bad_tokens, exclude_token_ids=[
    #     151643,    # <|endoftext|>
    #     151644,    # <|im_start|>
    # ])
    # agent_sampling_params.logits_processors = [exclude_func]
    agent_sampling_params.bad_words = ["<|endoftext|>", "<|im_start|>"]

    tokenizer = hf_tokenizer(config.agent.vl_model_path)
    processor = hf_processor(config.agent.vl_model_path)

    if multi_modal_inputs is not None:
        multi_modal_inputs = multi_modal_inputs.tolist()
    else:
        multi_modal_inputs = [{}] * len(vllm_inputs)

    env = ParallelEnv(config.agent, tokenizer, processor)
    env.reset(prompts, vllm_inputs, n=sampling_params.n)

    batch_size = len(vllm_inputs)
    vllm_input_list = []
    running_states = []
    running_action_masks = []
    running_attn_masks = []
    reward_tensor_list = []
    active_mask = []
    mm_input_list = []
    tool_call_cnt_list = []

    # interleaving inputs if sampling_params.n > 1
    for i in range(batch_size):
        for _ in range(sampling_params.n):
            vllm_input_list.append(vllm_inputs[i])
            prompt_ids = prompts.batch['input_ids'][i, :]
            running_states.append(prompt_ids)
            prompt_mask = prompts.batch['attention_mask'][i, :]
            running_action_masks.append(prompt_mask)
            running_attn_masks.append(prompt_mask)
            reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
            reward_tensor_list.append(reward_tensor)
            active_mask.append(True)
            mm_input_list.append(multi_modal_inputs[i])
            tool_call_cnt_list.append(0)

    max_total_length = config.prompt_length + config.response_length
    for step in range(config.agent.max_turns):
        print(f' [DEBUG 000] {step=}, total={batch_size}, n={sampling_params.n}, num_active={sum(active_mask)}')
        if sum(active_mask) == 0:
            break

        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
        active_vllm_inputs = [vinput for vinput, is_active in zip(vllm_input_list, active_mask) if is_active]
        actions = vllm_engine.generate(
            prompts=active_vllm_inputs,
            sampling_params=agent_sampling_params,
            use_tqdm=False
        )
        observations, rewards, dones, info = env.step(active_indices, actions)

        for idx, obs, act, rew, done in zip(active_indices, observations, actions, rewards, dones):
            # process response token ids
            response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64, device=running_states[idx].device)
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                vllm_input_list[idx]['prompt_token_ids'], 
                response_token_ids,
                tokenizer=tokenizer,
            )

            action_reward = torch.zeros_like(response_token_ids, dtype=torch.float, device=reward_tensor_list[idx].device)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
            reward_tensor_list[idx][-1] += rew

            action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=running_action_masks[idx].device)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], action_mask])

            # Ensure the last token is not obs
            if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                active_mask[idx] = False
                continue

            if done or step == config.agent.max_turns - 1:
                active_mask[idx] = False
                continue
            tool_call_cnt_list[idx] += 1

            # process obs tokens and images
            if 'prompt_token_ids_vllm' in obs.keys():
                obs_token_ids = obs['prompt_token_ids_vllm']
                vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                    vllm_input_list[idx]['prompt_token_ids'], 
                    obs_token_ids,
                    tokenizer=tokenizer,
                )

            if 'prompt_token_ids_model' in obs.keys():
                obs_token_ids = obs['prompt_token_ids_model'].to(running_states[idx].device)
                running_states[idx] = torch.cat([running_states[idx], obs_token_ids])

                obs_reward = torch.zeros(len(obs_token_ids), dtype=torch.float, device=reward_tensor_list[idx].device)
                reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

                obs_mask = torch.zeros(len(obs_token_ids), dtype=torch.int64, device=running_action_masks[idx].device)
                running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
                attn_mask = torch.ones(len(obs_token_ids), dtype=torch.int64, device=running_attn_masks[idx].device)
                running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

            mm_data = obs.get('multi_modal_data', {})
            if 'image' in mm_data.keys():
                if 'multi_modal_data' not in vllm_input_list[idx].keys():
                    vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                print(f' [DEBUG img] {idx=} before update {len(mm_data["image"])=}')
                vllm_input_list[idx]['multi_modal_data']['image'] += mm_data['image']
                print(f' [DEBUG img] {idx=} after update {len(vllm_input_list[idx]["multi_modal_data"]["image"])=}')

            mm_input = obs.get('multi_modal_inputs', {})
            if mm_input:
                print(f' [DEBUG img] {idx=} merge mm_input {mm_input_list[idx].keys()} + {mm_input.keys()}')
                mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], mm_input)

            if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                active_mask[idx] = False

    env.close()
    target_device = prompts.batch['input_ids'].device
    running_states = [state[: max_total_length] for state in running_states]
    state_tensor = pad_2d_list_to_length(running_states, tokenizer.pad_token_id, max_total_length).to(target_device)

    running_action_masks = [mask[: max_total_length] for mask in running_action_masks]
    action_mask_tensor = pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)

    running_attn_masks = [mask[: max_total_length] for mask in running_attn_masks]
    attn_mask_tensor = pad_2d_list_to_length(running_attn_masks, 0, max_total_length).to(target_device)

    if processor is not None and processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
        # For Qwen-VL: (n*bs, 3, seq_len)
        position_ids_list = [
            get_rope_index(
                processor,
                input_ids=state_tensor[i, :],
                image_grid_thw=mm_input_list[i].get("image_grid_thw", None),
                video_grid_thw=mm_input_list[i].get("video_grid_thw", None),
                second_per_grid_ts=mm_input_list[i].get("second_per_grid_ts", None),
                attention_mask=attn_mask_tensor[i, :],
            ) for i in range(batch_size * sampling_params.n)
        ]
        position_ids_tensor = torch.stack(position_ids_list, dim=0)
    else:
        # For LM: (n*bs, seq_len)
        position_ids_tensor = compute_position_id_with_mask(attn_mask_tensor)

    reward_tensor_list = [reward[: max_total_length] for reward in reward_tensor_list]
    reward_tensor = pad_2d_list_to_length(reward_tensor_list, 0.0, max_total_length).to(target_device)

    tool_call_tensor = torch.tensor(tool_call_cnt_list, dtype=torch.float32).to(target_device).unsqueeze(1)
    return DataProto.from_dict(
        tensors={
            "response": state_tensor[:, -config.response_length: ],
            "action_mask": action_mask_tensor,
            "attention_mask": attn_mask_tensor,
            "position_ids": position_ids_tensor,
            "env_reward": reward_tensor[:, -config.response_length: ],
            "tool_cnt": tool_call_tensor,
        },
        non_tensors={"multi_modal_inputs": mm_input_list} if processor is not None else None
    )


def execute_tool_call(sample, tokenizer=None, processor=None, pbar=None):
    action_string = sample.get('action', '')
    tool = sample.get('tool', None)

    # non-agent data
    if action_string == '' or tool is None:
        return {}, 0.0, True, {}

    tool_result, reward, done, info = tool.execute(action_string)

    # post-process
    if not tool_result:
        tool_result_info = {}

    elif isinstance(tool_result, str):
        # Format 1: text output
        obs_token_ids = tokenizer.encode(tool_result, add_special_tokens=False)
        tool_result_info = {
            "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
            "prompt_token_ids_model": torch.tensor(obs_token_ids),
        }

    elif isinstance(tool_result, list) and isinstance(tool_result[0], dict):
        # Format 2: [{"role": "...", "content": "..."}, ...]
        obs_token_ids = tokenizer.apply_chat_template(tool_result, add_generation_prompt=True, return_tensors='pt')[0]

        # NOTE: skip the sp (and the \n token that comes after it) added by Qwen tokenizer
        eos_start_idx = torch.nonzero(obs_token_ids == tokenizer.eos_token_id)
        if eos_start_idx.shape[0] > 0:
            eos_start_idx = eos_start_idx[0].item()
            obs_token_ids = obs_token_ids[eos_start_idx + 2 : ]
        else:
            raise ValueError(f"tool [{tool.name}] returned type List[str] output must be in openai/qwen format : {tool_result}")

        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids,
            "prompt_token_ids_model": obs_token_ids,
        }

    elif isinstance(tool_result, dict):
        # Format 3: {"prompt": "...", "chat": [{"role": "...", "content": "..."}, ...], "multi_modal_data": ...}
        prompt_str = tool_result.pop("prompt", "")
        chat_list = tool_result.pop("chat", [])

        if len(prompt_str) == 0 and len(chat_list) == 0:
            raise ValueError("Both prompt_str and chat_list are invalid")
        elif len(prompt_str) == 0 and len(chat_list) > 0:
            prompt_str = tokenizer.apply_chat_template(chat_list, add_generation_prompt=True, tokenize=False)
            prompt_str = _strip_system_block(prompt_str)

        prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(prompt_str, processor, **tool_result)
        obs_token_ids_vllm = tokenizer.encode(prompt_str_vllm, add_special_tokens=False, return_tensors='pt')[0]
        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids_vllm,
            "prompt_token_ids_model": obs_token_ids_model,
            **tool_result   # multi_modal_data
        }
        if mm_inputs:
            tool_result_info["multi_modal_inputs"] = mm_inputs

    else:
        raise ValueError(f"Invalid tool_result type: {type(tool_result)=} -- {tool_result}")

    if pbar is not None:
        pbar.update(1)
    return tool_result_info, reward, done, info


class ParallelEnv:
    """
    The interface is designed to be the similar to : https://github.com/openai/gym
    """
    def __init__(self, env_config, tokenizer, processor, **kwargs):
        self.config = env_config
        self.tokenizer = tokenizer
        self.processor = processor

        # type: List[ Dict[ Str, ToolBase subclasses ] ]
        self.tools = []

    def step(self, active_indices, actions):
        """
        Input:
        - actions: vllm.RequestOutput

        Output:
        - observations: List[Dict], content like {"prompt_token_ids": ..., "multi_modal_data": ...}, 
                multi_modal_data only appears when there are images/videos in obs
        - rewards: List[ float ].
                each time after an action being executed, procedure rewards can be assigned to 
                the last valid token of model outputs. This might be useful for ..., 
                e.g., invalid action, code execution error, format error,
                or video game envs where immediate feedback is available.
        - dones: List[ Boolean ]
        - infos: Dict, for debugging only
        """
        obs_list = [{}] * len(actions)
        reward_list = [0.0] * len(actions)
        done_list = []
        valid_indices = []
        real_indices = []
        valid_actions = []
        
        # 1. filtering valid actions
        for i, (idx, act) in enumerate(zip(active_indices, actions)):
            if act.outputs[0].finish_reason == 'length':
                done_list.append(True)
                continue

            if len(act.outputs[0].token_ids) == 0:
                done_list.append(True)
                continue

            done_list.append(False)
            real_indices.append(i)
            valid_indices.append(idx)
            valid_actions.append(act.outputs[0].text)

        agent_inputs = []
        for i, idx, action in zip(real_indices, valid_indices, valid_actions):
            agent_inputs.append(dict(
                idx=i,
                valid_idx=idx,
                action=action,
                tool=self.tools[idx],
            ))

        # 2. executing actions (sync or async)
        num_workers = min(self.config.concurrent_workers, len(valid_actions))
        pbar = tqdm(total=len(valid_actions), desc=f'Tool calling on {num_workers} workers') if self.config.show_tqdm else None
        if num_workers <= 1:
            for agi in agent_inputs:
                subidx = agi['idx']
                obs, reward, done, info = execute_tool_call(agi, self.tokenizer, self.processor, pbar=pbar)
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done
        else:
            partial_tool_func = partial(execute_tool_call, tokenizer=self.tokenizer, processor=self.processor, pbar=pbar)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                raw_outputs = list(executor.map(partial_tool_func, agent_inputs))
            for agi, raw in zip(agent_inputs, raw_outputs):
                obs, reward, done = raw[0], raw[1], raw[2]
                subidx = agi['idx']
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done

        return obs_list, reward_list, done_list, {}

    def reset(self, prompts, vllm_inputs, n=1, **kwargs):
        self.tools = []
        reset_output_list = []
        assert len(prompts) == len(vllm_inputs), f"{len(prompts)=}, {len(vllm_inputs)=}"

        num_agent, num_non_agent = 0, 0
        for i in range(len(prompts)):
            data_item = prompts[i]  # DataProtoItem
            tool_name = data_item.non_tensor_batch.pop(self.config.tool_name_key, '')
            raw_prompt = data_item.non_tensor_batch.pop('raw_prompt', None)

            vllm_input_item = vllm_inputs[i]   # {"prompt_token_ids": ..., "multi_modal_data": ...}
            multi_modal_data = vllm_input_item.get("multi_modal_data", None)
            origin_multi_modal_data = data_item.non_tensor_batch.pop("origin_multi_modal_data", None)
            for _ in range(n):
                if tool_name:
                    # init tools from config field `tool_name_key`
                    tool_fns = ToolBase.create(tool_name)
                    reset_output = tool_fns.reset(
                        raw_prompt=raw_prompt, 
                        multi_modal_data=multi_modal_data,
                        origin_multi_modal_data=origin_multi_modal_data,
                    )
                    self.tools.append(tool_fns)
                    reset_output_list.append(reset_output)
                    num_agent += 1
                else:
                    # non-agent data
                    self.tools.append(None)
                    reset_output_list.append(None)
                    num_non_agent += 1
        
        print(f' [DEBUG agent] {num_agent=}, {num_non_agent=}')
        return reset_output_list

    def close(self):
        self.tools = []
