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
Preprocess the Geometry3k dataset to parquet format
"""

import os
import datasets
import pandas as pd
import json
from PIL import Image
import io
from io import BytesIO

from verl.utils.hdfs_io import copy, makedirs
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/fs-computility/mabasic/yangminghao/data/MMInstruction/ArxivQA')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'chart'
    data_path = '/fs-computility/mabasic/yangminghao/data/MMInstruction/ArxivQA'
    env_name = "visual_toolbox_v5"
    sys_prompt = PROMPT.SYSTEM_PROMPT_V5
    instruction_prompt = PROMPT.USER_PROMPT_V5

    # dataset = datasets.load_dataset('json', os.path.join(data_path, 'dataset.jsonl'))

    jsonl_path = os.path.join(data_path, 'arxivqa_filter_stage1_reform.jsonl')
    jsonl_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            jsonl_data.append(json.loads(line))
    
    field_names = jsonl_data[0].keys()

    data_dict = {field: [] for field in field_names}
    for item in jsonl_data:
        for field in field_names:
            data_dict[field].append(item[field])
        
    raw_dataset = datasets.Dataset.from_dict(data_dict)


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            id = example.pop('id')
            problem = example.pop('new_question')
            prompt = problem + instruction_prompt
            answer = example.pop('new_answer')
            image_url = example.pop('image')
            images = []
            
            image_path = os.path.join("/fs-computility/mabasic/yangminghao/data/MMInstruction/ArxivQA", image_url)
            
            try:
                with Image.open(image_path) as img:
                    # Check image dimensions
                    width, height = img.size
                    if width < 28 or height < 28:
                        # Skip this example by returning None
                        # The dataset filtering will handle None returns later
                        print("Image too small, skipping:", image_path)
                        return None
                    
                    image_format = img.format
                    img_byte_arr = io.BytesIO()
                    # Save image to byte stream
                    img.save(img_byte_arr, format=image_format)
                    # Get byte stream content
                    img_byte_arr = img_byte_arr.getvalue()
                    img_bytes = img_byte_arr
                
                images.append({'bytes': img_bytes, 'path': image_path})
                
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "system",
                            "content": sys_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "images": images,
                    "ability": "vl_chart",
                    "env_name": env_name,
                    "reward_model": {
                        "style": "model",
                        "ground_truth": answer
                    },
                    "extra_info": {
                        'split': split,
                        'index': str(id),
                        'answer': answer,
                        "question": problem,
                    }
                }
                return data
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                return None

        return process_fn

    # First map with size checking
    dataset = raw_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)

    # Filter out None values (skipped examples)
    dataset = dataset.filter(lambda x: x is not None, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, 'ArxivQA_chart42k_visual_toolbox_v5.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)