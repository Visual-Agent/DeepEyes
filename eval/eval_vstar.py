import os
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import torch
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
import io
from openai import OpenAI
import requests
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://10.39.19.140:8000/v1', help='API URL')
parser.add_argument('--vstar_bench_path', type=str, default=None, help='Path to the V* benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

vstar_bench_path = args.vstar_bench_path
save_path = args.save_path
save_path = os.path.join(save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

instruction_prompt_system = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""
USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

instruction_prompt_before = """Question: {question}
Options: {options}
""" + USER_PROMPT_V2

user_prompt = USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# the following code is copied from qwen-vl-utils
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


def process(img_arg):
    img, test_path = img_arg
    img_path = os.path.join(test_path, img)
    anno_path = os.path.join(test_path, img.replace('.jpg', '.json'))
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    question = anno['question']
    options = anno['options']
    correct_answer = anno['options'][0]
    shuffle(options)
    
    option_str = "\n"
    for i in range(len(options)):
        option_str += abc_map[i + 1] + '. ' + options[i] + '\n'
    
    prompt = instruction_prompt_before.format(question=question, options=option_str)
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
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 10240,
                "stop": ["<|im_end|>\n".strip()],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
            
            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                action_list = eval(action_list)

                bbox_list = []
                cropped_pil_image_content_list = []

                bbox_str = action_list['arguments']['bbox_2d']
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
                user_msg = user_prompt

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                for cropped_pil_image_content in cropped_pil_image_content_list:
                    content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_msg})
                content_f.append({"type": "text", "text": "</tool_response>"})

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
                

    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info['image'] = img
    save_info['question'] = question
    save_info['answer'] = correct_answer
    save_info['answer_choice'] = chr(ord('A') + options.index(correct_answer))
    save_info['pred_ans'] = output_text
    save_info['pred_output'] = print_messages
    save_info['status'] = status
    return save_info


if __name__ == "__main__":
    test_types = ['direct_attributes', 'relative_position']

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}.jsonl"
        save_json = []
        test_path = os.path.join(vstar_bench_path, test_type)
        pool = multiprocessing.Pool(processes=args.num_workers)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(test_path)))
        image_args = [[img, test_path] for img in image_files]

        with tqdm(total=len(image_args), desc="Processing V* "+test_type) as pbar:
            for result in pool.imap(process, image_args):
                if result is not None:
                    save_json.append(result)
                    pbar.update(1)

        pool.close()
        pool.join()
    
        with open(os.path.join(save_path, save_name), 'w') as f:
            for item in save_json:
                f.write(json.dumps(item) + '\n')
