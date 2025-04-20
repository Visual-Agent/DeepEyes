import os
import json
import torch
from verl import DataProto


class RLLoggingBoardLogger:
    
    def __init__(
        self,
        root_log_dir: str,
        project_name: str,
        experiment_name: str
    ):
        self.save_path = os.path.join(
            root_log_dir, 
            project_name, 
            experiment_name
        )
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except:
            pass

    def log(
        self,
        data: dict,
        step: int,
        batch: DataProto,
        *args,
        **kwargs
    ):
        if 'tokenizer' not in kwargs:
            raise ValueError("Please provide a tokenizer.")
        
        tokenizer = kwargs['tokenizer']
        
        rm_response_list = kwargs['rm_response_list'] if 'rm_response_list' in kwargs else None
        with open(os.path.join(self.save_path, f"rollout_data_rank0.jsonl"), "a") as f:
            for i in range(len(batch)):
                data_item = batch[i]
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                rm_response = rm_response_list[i] if rm_response_list else None

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = tokenizer.decode(valid_prompt_ids)
                response_str = tokenizer.decode(valid_response_ids)
                response_tokens = [tokenizer.decode([token]) for token in valid_response_ids]
                cur_sample = {
                    "step": step,
                    "prompt": prompt_str,
                    "response": response_str,
                    "response_tokens": response_tokens,
                    "logprobs": data_item.batch['old_log_probs'][:valid_response_length].cpu().tolist(),
                    # "ref_logprobs": data_item.batch['ref_log_prob'][:valid_response_length].cpu().tolist(),
                    # "values": data_item.batch['values'][:valid_response_length].cpu().tolist(),
                    "token_rewards": data_item.batch['token_level_rewards'][:valid_response_length].cpu().tolist(),     # with KL penalty
                    "reward": data_item.batch['token_level_scores'][:valid_response_length].cpu().sum().item(),         # without KL penalty"
                }
                
                if "ground_truth" in data_item.non_tensor_batch['reward_model']:
                    cur_sample["ground_truth"] = data_item.non_tensor_batch['reward_model']["ground_truth"]

                if "values" in data_item.batch:
                    cur_sample['values'] = data_item.batch['values'][:valid_response_length].cpu().tolist()
                
                if rm_response is not None:
                    cur_sample['rm_response'] =rm_response
                
                f.write(f"{json.dumps(cur_sample, ensure_ascii=False)}\n")