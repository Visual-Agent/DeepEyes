import re
import os
import datasets
from tqdm import tqdm
import json
import re
import time
import sys
import json
import random
import argparse
from verl.workers.agent.envs.collab_code.system_prompt import USER_SP, ASSISTANT_SP, CODE_INITIAL_PROMPT


def extract_code_content(content):
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    return matches


def make_map_fn(split, dataset_name):

    def process_fn(example, idx):
        
        core = example['collab_divide']['Core Instruction']
        hidden = '\n'.join([f'{i+1}: {supp}' for i, supp in enumerate(example['collab_divide']['Supplementary Instructions'])])
        
        starter_code = extract_code_content(example['query'])[0].strip()
        user_prompt = CODE_INITIAL_PROMPT.format(question=core, code=starter_code)
        ground_truth = example['completion']
        test_cases = f"{example['test']}\n\ncheck({example['entry_point'].strip()})"

        data = {
            "data_source": dataset_name,
            "env_name": dataset_name,
            "prompt": [
            {
                "role": "system",
                "content": ASSISTANT_SP['code_v3']
            },
            {
                "role": "user",
                "content": user_prompt
            }],
            "ability": "chat",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                'split': split,
                'index': idx
            },
            "env_info": {
                'hidden': hidden,
                'test_cases': test_cases
            }
        }
        return data

    return process_fn


def construct_rl_prompt(args):
    dataset = datasets.load_dataset('json', data_files=args.local_train_path)
    dataset = dataset.map(function=make_map_fn('train', args.dataset_name), with_indices=True)
    dataset['train'].to_parquet(os.path.join(args.output_dir, 'train_spv3.parquet'))
    
    dataset = datasets.load_dataset('json', data_files=args.local_test_path)
    dataset = dataset.map(function=make_map_fn('test', args.dataset_name), with_indices=True)
    dataset['train'].to_parquet(os.path.join(args.output_dir, 'test_spv3.parquet'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_train_path', default='./data/collab_code/split_prompt/LeetCodeDataset-v0.3.0-train-claude.jsonl')
    parser.add_argument('--local_test_path', default='./data/collab_code/split_prompt/LeetCodeDataset-v0.3.0-test-claude.jsonl')
    parser.add_argument('--output_dir', default='./data/collab_code/LeetCodeDataset-v0.3.0')
    parser.add_argument('--dataset_name', default='collab_code')

    args = parser.parse_args()

    construct_rl_prompt(args)
