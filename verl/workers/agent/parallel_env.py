import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.agent.tool_envs import ToolBase

def _concat_vllm_input(prompt_token_ids, response_token_ids):
    if isinstance(prompt_token_ids, torch.Tensor):
        return torch.cat([
            prompt_token_ids,
            response_token_ids.to(prompt_token_ids.device),
        ], dim=-1)
    else:
        return np.concatenate([
            prompt_token_ids,
            response_token_ids.cpu().numpy(),
        ], axis=-1)

def agent_rollout_loop(config, tokenizer, vllm_engine, vllm_inputs, prompts, sampling_params):
    agent_sampling_params = sampling_params.clone()
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.spaces_between_special_tokens = False
    agent_sampling_params.n = 1
    agent_sampling_params.include_stop_str_in_output = True
    max_generated_tokens = min(config.agent.single_response_max_tokens, config.response_length)
    agent_sampling_params.max_tokens = max_generated_tokens

    # NOTE: to prevent oov issue for qwen base
    agent_sampling_params.top_p = 0.99

    # support custom stop specified in dataset, like </search>, ```, etc.
    custom_stop = config.agent.custom_stop
    if custom_stop:
        if isinstance(custom_stop, str):
            custom_stop = [custom_stop]
        prev_stop = sampling_params.stop if sampling_params.stop else []
        agent_sampling_params.stop = prev_stop + custom_stop
    env = ParallelEnv(config.agent, tokenizer)
    env.reset(prompts, n=sampling_params.n)
    batch_size = len(vllm_inputs)
    vllm_input_list = []
    running_states = []
    running_action_masks = []
    reward_tensor_list = []
    active_mask = []

    # interleaving inputs if sampling_params.n > 1
    for i in range(batch_size):
        for _ in range(sampling_params.n):
            vllm_input_list.append(vllm_inputs[i])
            prompt_ids = prompts.batch['input_ids'][i, :]
            running_states.append(prompt_ids)
            prompt_mask = prompts.batch['attention_mask'][i, :]
            running_action_masks.append(prompt_mask)
            reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
            reward_tensor_list.append(reward_tensor)
            active_mask.append(True)

    for step in range(config.agent.max_turns):
        print(f' [DEBUG 000] {step=}, total={batch_size}, n={sampling_params.n}, num_active={sum(active_mask)}')
        if sum(active_mask) == 0:
            break

        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
        active_vllm_input = [vinput for vinput, is_active in zip(vllm_input_list, active_mask) if is_active]
        actions = vllm_engine.generate(
            prompts=active_vllm_input,
            sampling_params=agent_sampling_params,
            use_tqdm=False
        )
        observations, rewards, dones, info = env.step(actions)
        # obs_token_ids = [obs['prompt_token_ids'] for obs in observations]
        # # TODO: support mm input in obs!!!
        # multi_modal_data = [obs['multi_modal_data'] for obs in observations]

        # for idx, obs, mm, act, rew, done in zip(active_indices, obs_token_ids, multi_modal_data, actions, rewards, dones):
        for idx, obs, act, rew, done in zip(active_indices, observations, actions, rewards, dones):
            # process response token ids
            response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64, device=running_states[idx].device)
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(vllm_input_list[idx]['prompt_token_ids'], response_token_ids)

            action_reward = torch.zeros_like(response_token_ids, dtype=torch.float, device=reward_tensor_list[idx].device)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
            reward_tensor_list[idx][-1] += rew

            action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=running_action_masks[idx].device)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])
            # print(f' [DEBUG resp] resp_size={response_token_ids.shape[-1]}, total_size={running_states[idx].shape[-1]}')

            # NOTE: Ensure the last tokens are not obs
            if step == config.agent.max_turns - 1:
                continue

            # process obs tokens and images
            if 'prompt_token_ids' in obs:
                obs_token_ids = obs['prompt_token_ids'].to(running_states[idx].device)
                # if obs gets too long, truncated obs will be fed into the model
                if len(obs_token_ids) > config.agent.single_obs_max_length:
                    print("[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG")
                    obs_token_ids = obs_token_ids[:config.agent.single_obs_max_length]

                running_states[idx] = torch.cat([running_states[idx], obs_token_ids])
                vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(vllm_input_list[idx]['prompt_token_ids'], obs_token_ids)

                obs_reward = torch.zeros(len(obs_token_ids), dtype=torch.float, device=reward_tensor_list[idx].device)
                reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

                obs_mask = torch.zeros(len(obs_token_ids), dtype=torch.int64, device=running_action_masks[idx].device)
                running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
                # print(f' [DEBUG obs] obs_size={len(obs_token_ids)}, total_size={running_states[idx].shape[-1]}')

            # TODO: check whether the truncation is correct here
            mm = obs.get('multi_modal_data', {})
            if 'image' in mm.keys():
                if 'multi_modal_data' not in vllm_input_list[idx]:
                    vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                vllm_input_list[idx]['multi_modal_data']['image'] += mm['image']
                print(f' [DEBUG img] mm_image_size={len(mm["image"])}')

            if done or running_states[idx].shape[-1] >= config.prompt_length + config.response_length:
                active_mask[idx] = False

    env.close()
    # running_response_string = tokenizer.batch_decode(
    #     running_states,
    #     skip_special_tokens=False,
    # )
    # print(f" [DEBUG response] {running_response_string[:3]}=")
    # # print(f' [DEBUG state] {running_states[0]=}')
    # print(f" [DEBUG mask] {running_action_masks[:3]}=")
    max_total_length = config.prompt_length + config.response_length
    target_device = prompts.batch['input_ids'].device
    running_states = [state[: max_total_length] for state in running_states]
    state_tensor = pad_2d_list_to_length(running_states, tokenizer.pad_token_id, max_total_length).to(target_device)
    running_action_masks = [mask[: max_total_length] for mask in running_action_masks]
    action_mask_tensor = pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)
    reward_tensor_list = [reward[: max_total_length] for reward in reward_tensor_list]
    reward_tensor = pad_2d_list_to_length(reward_tensor_list, 0.0, max_total_length).to(target_device)
    return state_tensor[:, -config.response_length: ], action_mask_tensor, reward_tensor[:, -config.response_length: ]


def execute_tool_call(sample, tokenizer=None, pbar=None):
    action_string = sample.get('action', '')
    tool = sample.get('tool', None)
    agent_meta = sample.get('agent_meta', {})
    # print(f' [DEBUG rag input] {sample["idx"]=}, {action_string=}, {tool=}, agent_meta={agent_meta}')
    # print(f' [DEBUG frozenlake input] {sample["idx"]=}, {action_string=}, {tool=}, agent_meta={agent_meta}')
    # non-agent data
    if action_string == '' or tool is None:
        return {}, 0.0, True, {}

    tool_result, reward, done, info = tool.execute(action_string, meta=agent_meta)

    # post-process
    if not tool_result:
        # print(f' [WARNING] {action_string=} is not valid for tool={tool.name}')
        tool_result_info = {}
    elif isinstance(tool_result, dict):
        # 1. tool return: {"prompt_token_ids: ..., "multi_modal_data": ...}
        # TODO: Re-generate prompt_token_ids according to multi_modal_data
        tool_result_info = tool_result

    elif isinstance(tool_result, str) and len(tool_result) > 0:
        # 2. tool return str
        obs_token_ids = tokenizer.encode(tool_result)
        tool_result_info = {"prompt_token_ids": torch.tensor(obs_token_ids)}
        # print(f' [DEBUG rag output] {tool_result=}, {len(obs_token_ids)=}')

    elif isinstance(tool_result, list) and len(tool_result) > 0 and isinstance(tool_result[0], str):
        # 3. List[str], treated as openai/qwen style output
        msg_list = [{"role": "tool", "content": res} for res in tool_result]
        obs_token_ids = tokenizer.apply_chat_template(msg_list, add_generation_prompt=True, return_tensors='pt')[0]
        # VERY UGLY: remove the system prompt added by qwen tokenizer
        eos_start_idx = torch.nonzero(obs_token_ids == self.tokenizer.eos_token_id)
        if eos_start_idx.shape[0] > 0:
            eos_start_idx = eos_start_idx[0].item()
            obs_token_ids = obs_token_ids[eos_start_idx + 1 : ]
        else:
            raise ValueError(f"tool [{tool.name}] returned type List[str] output must be in openai/qwen format : {tool_result}")
        if not isinstance(obs_token_ids, torch.Tensor):
            obs_token_ids = torch.tensor(obs_token_ids)
        tool_result_info = {"prompt_token_ids": obs_token_ids}
    else:
        raise ValueError(f"Invalid tool_result type: {type(tool_result)=} -- {tool_result}")

    if pbar is not None:
        pbar.update(1)
    return tool_result_info, reward, done, info


class ParallelEnv:
    """
    The interface intentionally designed to be the similar to : https://github.com/openai/gym
    Hope this could be easier to use for RLers.
    """
    def __init__(self, env_config, tokenizer, **kwargs):
        self.config = env_config
        self.tokenizer = tokenizer
        # type: List[ Dict[ Str, ToolBase subclasses ] ]
        self.tools = []
        self.agent_meta = []

    def step(self, actions):
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
        valid_actions = []
        
        # 1. filtering valid actions
        for idx, act in enumerate(actions):
            # print(f' [DEBUG vllm output] {idx=}, {act.outputs[0].finish_reason=}, {act.outputs[0].stop_reason=}')
            if act.outputs[0].finish_reason == 'length':
                done_list.append(True)
                continue
            done_list.append(False)
            valid_indices.append(idx)
            valid_actions.append(act.outputs[0].text)

        agent_inputs = []
        for idx, action in zip(valid_indices, valid_actions):
            agent_inputs.append(dict(
                idx=idx,
                action=action,
                tool=self.tools[idx],
                agent_meta=self.agent_meta[idx],
            ))

        # 2. executing actions (sync or async)
        num_workers = min(self.config.concurrent_workers, len(valid_actions))
        pbar = tqdm(total=len(valid_actions), desc=f'Executing tool call on {num_workers} workers') if self.config.show_tqdm else None
        if num_workers <= 1:
            for agi in agent_inputs:
                subidx = agi['idx']
                obs, reward, done, info = execute_tool_call(agi, self.tokenizer, pbar=pbar)
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done
        else:
            partial_tool_func = partial(execute_tool_call, tokenizer=self.tokenizer, pbar=pbar)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                raw_outputs = list(executor.map(partial_tool_func, agent_inputs))
            for agi, raw in zip(agent_inputs, raw_outputs):
                obs, reward, done = raw[0], raw[1], raw[2]
                subidx = agi['idx']
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done

        return obs_list, reward_list, done_list, {}

    def reset(self, prompts, n=1):
        self.tools = []
        self.agent_meta = []
        print(f' [DEBUG reset] {prompts.batch.keys()=}, {prompts.non_tensor_batch.keys()=}, {prompts.meta_info.keys()=}')

        reset_output_list = []
        for i in range(len(prompts)):
            for _ in range(n):
                data_item = prompts[i]  # DataProtoItem

                if self.config.tool_meta_key:
                    tool_meta = data_item.non_tensor_batch.get(self.config.tool_meta_key, None)
                else:
                    tool_meta = None
                self.agent_meta.append(tool_meta)

                tool_name = data_item.non_tensor_batch.get(self.config.tool_name_key, '')
                if tool_name:
                    # init tools from config field `tool_name_key`
                    tool_fns = ToolBase.create(tool_name)
                    reset_output = tool_fns.reset(tool_meta)
                    self.tools.append(tool_fns)
                    reset_output_list.append(reset_output)
                else:
                    # non-agent data
                    self.tools.append(None)
                    reset_output_list.append(None)

        # NOTE: pop agent keys to prevent batch_size mismatch when n>1
        if self.config.tool_name_key and self.config.tool_name_key in prompts.non_tensor_batch.keys():
            prompts.non_tensor_batch.pop(self.config.tool_name_key)
            print(f' [DEBUG tools] non_gensor_batch pop key={self.config.tool_name_key}')
        if self.config.tool_meta_key and self.config.tool_meta_key in prompts.non_tensor_batch.keys():
            prompts.non_tensor_batch.pop(self.config.tool_meta_key)
            print(f' [DEBUG tools] non_gensor_batch pop key={self.config.tool_meta_key}')
        print(f' [DEBUG tools] {len(self.tools)=}, {len(self.agent_meta)=}')
        return reset_output_list

    def close(self):
        self.tools = []
        self.agent_meta = []
