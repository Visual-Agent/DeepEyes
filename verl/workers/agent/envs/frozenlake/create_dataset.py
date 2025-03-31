"""
Preprocess dataset for frozenlake task 
"""

import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets

from frozenlake import FrozenLakeTool

templates = {
    'qwen-instruct': '{prompt}\nAlways output: <think> [Your thoughts] </think> <action> [your action] </action> with no extra text. Strictly follow this format.',
}

intro = """You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G).

Symbols:
F Frozen | H Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (H).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Action Space:
Choose one action from the following four actions.
<action> left </action> | <action> down </action> | <action> right </action> | <action> up </action>

Rewards:
Fall into hole: 0
Reach goal: +1.0

Episode End:
The episode ends if the following happens:
1. You moves into a hole.
2. You reaches the goal. 

{observation}
Decide the next action:\
"""

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--env", type=str, default="frozenlake", help="Environment name (default: 'frozenlake').")
    parser.add_argument("--seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/frozenlake", help="Output file to save the trajectories (default: 'data/frozenlake').")
    parser.add_argument("--train_size", type=int, default=3000, help="Number of trajectories to generate (default: 3000).")
    parser.add_argument("--test_size", type=int, default=100, help="Number of trajectories to generate (default: 100).")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])
    parser.add_argument("--use_mm", action="store_true", default=False, help="Return the rgb_array observation")
    args = parser.parse_args()
    
    assert args.env == "frozenlake", "Unsupported environment: {args.env}"
    os.makedirs(args.output, exist_ok=True)
    data_source = args.env
    
    size, p = os.environ.get("SIZE"), os.environ.get("P")
    size, p = int(size), float(p)
    use_mm = args.use_mm


    # Generate instruction
    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    instructions = []
    for seed in seeds:
        env = FrozenLakeTool(size=size, p=p, seed=seed, use_mm=use_mm, _name=None, _desc=None, _params=None)
        observation = env.render(train=False)
        instruction = intro.format(observation=observation)
        instructions.append(instruction)
    

    def _create_instance(idx, instruction):
        prompt_formatted = templates[args.prefix].format(prompt=instruction)
        print(prompt_formatted)

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "env_name": "frozenlake",
            "extra_info": {"split": "train", "index": idx}
        }
    train_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i]) for i in range(args.train_size)])
    test_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i]) for i in range(args.train_size, args.train_size + args.test_size)])


    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn

    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))

if __name__ == "__main__":
    main()