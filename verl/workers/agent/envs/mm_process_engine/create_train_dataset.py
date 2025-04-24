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
    parser.add_argument('--local_dir', default='/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'vstar'
    data_path = '/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/ziwei/distilled'
    env_name = "visual_toolbox_v3"
    instruction_prompt = PROMPT.USER_PROMPT_V3

    # dataset = datasets.load_dataset('json', os.path.join(data_path, 'dataset.jsonl'))

    jsonl_path = os.path.join(data_path, 'gqa_6k.jsonl')
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
            # print (example)
            
            problem = example.pop('question')
            prompt = '<image>\n' + '\nQuestion:\n' + problem + instruction_prompt
            answer = example.pop('answer')
            image_url = example.pop('image')
            images = []
            target_size = (224, 224)

            image_path = os.path.join("/fs-computility/mabasic/yangminghao/data", image_url)
            # with open(image_path, 'rb') as img_file:
            #     img_bytes = img_file.read()
            with Image.open(image_path) as img:
                image_format = img.format
                img_byte_arr = io.BytesIO()
                # 将图片保存到字节流中
                img.save(img_byte_arr, format=image_format)
                # 获取字节流的内容
                img_byte_arr = img_byte_arr.getvalue()
                img_bytes = img_byte_arr
            images.append({'bytes': img_bytes, 'path': image_path})
            

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": "",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,
                "ability": "vl",
                "env_name": env_name,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                }
            }
            return data

        return process_fn

    dataset = raw_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, 'vl_agent_V3_train_gqa_6k_new.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)