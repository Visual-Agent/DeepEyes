import os
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import random
import time
import math


from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image
import base64
import io
from openai import OpenAI
import requests

openai_api_key = ""
# openai_api_base = "http://localhost:8000/v1"
openai_api_base = ""

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
response = requests.get(f"{openai_api_base}/models")
models = response.json()
model_name = models['data'][0]['id']

vstar_bench_path = '/cpfs/user/honglingyi/DATA/LLM/Vstar/vstar_bench'
test_types = ['direct_attributes', 'relative_position']
per_type_acc = {}
for test_type in test_types:
    per_type_acc[test_type] = []
all_acc = []
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

# instruction_prompt_system = "You are a helpful assistant. The user will ask you a question and you as the assistant solve it. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively."

# instruction_prompt_before = """Question: {question}

# If the images provided above are not sufficient to answer the user's question, please generate a grounding for the region of interest in the image. The grounding region should incorporate the information required to address the above question.

# The output grounding should be in JSON format:
# ```json
# [
#     {{"bbox_2d": [x1, y1, x2, y2], "label": "label name"}}
# ]
# ```
# The zoomed-in images of your grounding results will be provided in next turn.

# Otherwise, please put your final answer in <answer> </answer> tags.
# """

# user_prompt = "Here is the zoomed in image for your grounding region {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise generate a new grouding in JSON format."


instruction_prompt_system = ""

instruction_prompt_before = """Question: {question}

If the images provided above are not sufficient to answer the user's question, please generate grouding results in JSON format:
```json
[
    {{"bbox_2d": [x1, y1, x2, y2], "label": "label name"}}
]
```
The zoomed-in images of your grounding results will be provided in next turn.

Otherwise, please put your final answer in <answer> </answer> tags.
"""

user_prompt = """If the images provided above are not sufficient to answer the user's question, please generate grouding results in JSON format:
```json
[
    {"bbox_2d": [x1, y1, x2, y2], "label": "label name"}
]
```
The zoomed-in images of your grounding results will be provided in next turn.

Otherwise, please put your final answer in <answer> </answer> tags.
"""

def encode_image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_pil_image_to_base64(pil_image):
    """将图片转换为base64编码"""
    # return base64.b64encode(pil_image.read()).decode('utf-8')
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # 这里假设保存为PNG格式，你可以根据需要修改
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fix_and_parse_json(json_str):
    # 尝试简单修复引号问题
    fixed_str = json_str.replace("'", "\"")
    try:
        # 解析修复后的字符串
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        # 如果解析失败，可能需要更复杂的修复
        import re
        # 匹配所有 { 和 } 之间的内容，尝试修复引号
        pattern = re.compile(r'({.*?})')
        matches = pattern.findall(fixed_str)
        for match in matches:
            # 去掉多余的引号
            new_match = match.replace('"{"', '{').replace('"}"', '}')
            fixed_str = fixed_str.replace(match, new_match)
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError as e:
            print(f"仍然无法解析: {e}")
            return None

# test_types = [test_types[0]]
for test_type in test_types:
    save_json = []
    test_path = os.path.join(vstar_bench_path, test_type)
    image_files = list(filter(lambda file: '.json' not in file, os.listdir(test_path)))
    # image_files = image_files[27:]
    for img in tqdm(image_files):
    # for img in image_files:
        img_path = os.path.join(test_path, img)
        anno_path = os.path.join(test_path, img.replace('.jpg', '.json'))
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        question = anno['question']
        options = anno['options']

        prompt = instruction_prompt_before.format(question=question)
        pil_img = Image.open(img_path)

        base64_image = encode_image_to_base64(img_path)

        messages = [
            {
                "role": "system",
                "content": instruction_prompt_system,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        print_messages = [
            {
                "role": "system",
                "content": instruction_prompt_system,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_message = messages

        response_message = ""

        status = 'success'
        try_count = 0
        turn_idx = 0
        try:
            while '</answer>' not in response_message:
                if '</answer>' in response_message and '<answer>' in response_message:
                    break

                if try_count > 10:
                    # status = 'error'
                    break

                params = {
                    "model": model_name,  # 或VLLM服务中的任何模型名称
                    "messages": chat_message,
                    "temperature": 0.0,
                    "max_tokens": 20480,
                    "stop": ["<|im_end|>\n".strip()],
                }
                print('print message:   ',print_messages)
                response = client.chat.completions.create(**params)
                response_message = response.choices[0].message.content
                
                print('response message:   ',response_message)
                if '```' in response_message:
                    action_list = response_message.split('```json')[1].split('```')[0].strip()
                    action_list = eval(action_list)

                    bbox_list = []
                    cropped_pil_image_content_list = []

                    for _action_list in action_list:
                        bbox_str = _action_list['bbox_2d']
                        bbox = bbox_str
                        left, top, right, bottom = bbox
                        cropped_image = pil_img.crop((left, top, right, bottom))
                        new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                        cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                        cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                        bbox_list.append(bbox)
                        cropped_pil_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}}
                        cropped_pil_image_content_list.append(cropped_pil_image_content)

                    if len(bbox_list) == 1:
                        bbox_list = bbox_list[0]
                    # user_msg = user_prompt.format(bbox_list)
                    user_msg = user_prompt

                    content_f = []
                    for cropped_pil_image_content in cropped_pil_image_content_list:
                        content_f.append(cropped_pil_image_content)
                    content_f.append({"type": "text", "text": user_msg})
                    
                    _message =[
                        {
                            "role": "assistant",
                            "content": response_message,
                        },
                        {
                            "role": "user",
                            "content": content_f,
                        }
                    ]

                    chat_message.extend(_message)
                    
                
                    p_message =[
                        {
                            "role": "assistant",
                            "content": response_message,
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                                {"type": "text", "text": user_msg},
                            ],
                        }
                    ]
                    print_messages.extend(p_message)
                    turn_idx += 1
                    print ('Next turn: ', turn_idx)
                else:
                    p_message =[
                        {
                            "role": "assistant",
                            "content": response_message,
                        }
                    ]
                    print_messages.extend(p_message)


                try_count += 1
        except Exception as e:
            print(f"Error!!!!", e)
            status = 'error'
                    


        print ("endddddd")
        if '</answer>' in response_message and '<answer>' in response_message:
            output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
        else:
            output_text = 'ERROR'

        save_info = {}
        save_info['image'] = img
        save_info['question'] = question
        save_info['answer'] = anno['options'][0]
        save_info['pred_ans'] = output_text
        save_info['pred_output'] = print_messages
        save_info['status'] = status

        save_json.append(save_info)

    with open('result_'+test_type+'.jsonl', 'w') as f:
        for item in save_json:
            f.write(json.dumps(item) + '\n')



