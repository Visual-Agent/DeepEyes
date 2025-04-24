import re
import os
import datasets
from tqdm import tqdm
import json
import re
import time
import io
import json
import random
import argparse
from io import BytesIO
from PIL import Image
import base64


query_text = """Question: {question}
If the images provided above are not sufficient to answer the user's question, please generate grouding results in JSON format:
```json
[
    {{"bbox_2d": [x1, y1, x2, y2], "label": "label name"}}
]
```
The zoomed-in images of your grounding results will be provided in next turn.

Otherwise, please put your final answer in <answer> </answer> tags.
"""

baseline_query = "{question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

image_path_prefix = "/cpfs/user/honglingyi/DATA/LLM/Vstar"


def make_map_fn(split, data_source, env_name):

    def process_fn(data, idx):
        
        question = data['conversations'][0]['value'].split('<object>.\n')[-1]
        gt = data['conversations'][1]['value']
        image_path = os.path.join(image_path_prefix, data['image'])
        
        with Image.open(image_path) as img:
            image_format = img.format
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=image_format)
            img_bytes = img_byte_arr.getvalue()
        
        images = [{'bytes': img_bytes, 'path': None}]

        data = {
            "data_source": data_source,
            "env_name": env_name,
            "prompt": [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": "<image>\n" + query_text.format(question=question),
                    # "content": "<image>\n" + baseline_query.format(question=question),
                },
            ],
            'images': images,
            "ability": "vl",
            "reward_model": {
                "style": "model",
                "ground_truth": gt
            },
            "extra_info": {
                'split': split,
                'id': idx,
                'question': question
            }
        }
        return data

    return process_fn


def construct_rl_prompt(args):
    dataset = datasets.load_dataset('json', data_files=args.local_jsonl_path, split='train')
    # only choose sample that answer correctly with the 2turn setting.
    dataset = dataset.filter(lambda d: d.get("result", True))
    total_size = len(dataset)
    proposed_val_size = int(total_size * args.val_ratio)
    # proposed_val_size = min(args.val_size, int(total_size * args.val_ratio))
    # assert total_size > proposed_val_size

    split_dataset = dataset.train_test_split(test_size=proposed_val_size, seed=42, shuffle=True)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    train_dataset = train_dataset.map(function=make_map_fn('train', args.data_source, args.env_name),
                                      remove_columns=dataset.column_names,
                                      with_indices=True)
    train_dataset.to_parquet(os.path.join(args.output_dir, f'train_{args.note}.parquet'))
    
    val_dataset = val_dataset.map(function=make_map_fn('test', args.data_source, args.env_name),
                                  remove_columns=dataset.column_names,
                                  with_indices=True)
    val_dataset.to_parquet(os.path.join(args.output_dir, f'val_{args.note}.parquet'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_jsonl_path', default='./data/vlagent/distilled/vaw_attribute_1t_fail.json')
    parser.add_argument('--output_dir', default='./data/vlagent/parquet')
    parser.add_argument('--note', default='vaw_attribute_1t_fail')
    # parser.add_argument('--val_size', default=1000)
    parser.add_argument('--val_ratio', default=0.1)
    parser.add_argument('--data_source', default='vl_agent')
    parser.add_argument('--env_name', default='vl_agent_v3')

    args = parser.parse_args()

    construct_rl_prompt(args)
