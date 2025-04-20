import requests
import json
import argparse
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
import time
import pandas as pd
import base64
import requests
import os
from openai import OpenAI
from io import BytesIO
from PIL import Image
from math import floor, ceil


vlm_api_key = "zzw-114514"
vlm_api_base = "http://10.39.17.198:8000/v1"
vlm_client = OpenAI(
    api_key=vlm_api_key,
    base_url=vlm_api_base
)
models = vlm_client.models.list()
vlm_model = models.data[0].id
image_path_prefix = "/cpfs/user/honglingyi/DATA/LLM/Vstar"

llm_api_key = "zzw-114514"
llm_api_base = "http://10.39.23.170:8000/v1"
llm_client = OpenAI(
    api_key=llm_api_key,
    base_url=llm_api_base
)
models = llm_client.models.list()
llm_model = models.data[0].id

second_turn_trigger = "I need to look carefully at this image. Can you provide the cropped images related to this question?"


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

def get_judge_prompt(question, predict_str, ground_truth):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    return f'{demo_prompt}{test_prompt}'


def maybe_resize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if height < 28 or width < 28:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        ratio = 28 / min(height, width)
        new_half_width = ceil(width * ratio * 0.5)
        new_half_height = ceil(height * ratio * 0.5)
        new_x1 = floor(center_x - new_half_width)
        new_x2 = ceil(center_x + new_half_width)
        new_y1 = floor(center_y - new_half_height)
        new_y2 = ceil(center_y + new_half_height)
        return [new_x1, new_y1, new_x2, new_y2]
    return [x1, y1, x2, y2]


def encode_base64_content_from_local_path(image_path, bbox=None):
    image = Image.open(image_path).convert("RGB")
    if bbox:
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        bbox = maybe_resize_bbox(bbox)
        image = image.crop(bbox)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

# Single-image input inference
def call_vlm(prompt, image_path, bboxs=None):
    image_path = os.path.join(image_path_prefix, image_path)
    ## Use base64 encoded image in the payload
    image_base64 = encode_base64_content_from_local_path(image_path)
    convs = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]},]
    if bboxs:
        convs.append({"role": "assistant", "content": second_turn_trigger})
        content = [{"type": "text", "text": "Sure."}]
        for bbox in bboxs:
            image_base64_cropped = encode_base64_content_from_local_path(image_path, bbox=bbox)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_cropped}"}})
        convs.append({"role": "user", "content": content})
    chat_completion_from_base64 = vlm_client.chat.completions.create(
        messages=convs,
        model=vlm_model,
        # max_completion_tokens=64,
    )
    result = chat_completion_from_base64.choices[0].message.content.strip()
    return result

def judge_response(prompt):
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=llm_model,
    )
    response = chat_completion.choices[0].message.content.strip()
    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    return acc_reward == 1.0

def process_prompt_1turn(data):
    max_retries = 100
    retry_delay = 3
    
    prompt = data['conversations'][0]['value'].split('<object>.\n')[-1]
    gt = data['conversations'][1]['value']
    image_path = data['image']
    
    if len(data['target_instances']) == 0:
        return None
    
    save_data = data
    for attempt in range(max_retries):
        try:
            prediction = call_vlm(prompt, image_path, bboxs=None)
            save_data['prediction'] = prediction
            judge_prompt = get_judge_prompt(prompt, prediction, gt)
            save_data['result'] = judge_response(judge_prompt)
            save_data['distill_status'] = 'success'
            return save_data
        except Exception as e:
            if attempt < max_retries - 1:
                print(e)
                print(f"第{attempt+1}次重试: {prompt[:50]}...")
                time.sleep(retry_delay)
                continue
            save_data['distill_status'] = 'fail'
            return save_data

def process_prompt_2turn(data):
    max_retries = 100
    retry_delay = 3
    
    if not data['result']:
        return None
    
    prompt = data['conversations'][0]['value'].split('<object>.\n')[-1]
    gt = data['conversations'][1]['value']
    image_path = data['image']
    
    if len(data['target_instances']) == 0:
        return None
    target_names = [t['name'] for t in data['target_instances']]
    bboxs = [t['bbox'] for t in data['target_instances']]
    
    save_data = data
    for attempt in range(max_retries):
        try:
            prediction = call_vlm(prompt, image_path, bboxs=bboxs)
            save_data['prediction'] = prediction
            judge_prompt = get_judge_prompt(prompt, prediction, gt)
            save_data['result'] = judge_response(judge_prompt)
            save_data['distill_status'] = 'success'
            return save_data
        except Exception as e:
            if attempt < max_retries - 1:
                print(e)
                print(f"第{attempt+1}次重试: {prompt[:50]}...")
                time.sleep(retry_delay)
                continue
            save_data['distill_status'] = 'fail'
            return save_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="./data/vlagent/seal_vqa_data/spatial_relation_data.json")
    parser.add_argument('--output', type=str, default="./data/vlagent/distilled/spatial_relation_raw.json")
    parser.add_argument('--processes', type=int, default=cpu_count())
    parser.add_argument('--multi_turn', action='store_true', default=False)
    parser.add_argument('--jsonl', action='store_true', default=False)
    args = parser.parse_args()

    # 读取输入文件
    print(f"正在读取文件: {args.input}")
    with open(args.input, 'r') as f:
        if args.jsonl:
            datas = [json.loads(line) for line in f]
        else:
            datas = json.load(f)
    print(f"成功读取 {len(datas)} 条data")
    
    process_prompt = process_prompt_1turn if not args.multi_turn else process_prompt_2turn

    # 创建进程池
    with open(args.output, 'a+') as write_f:
        with Pool(processes=args.processes) as pool:
            results = []
            with tqdm(total=len(datas), desc="处理进度") as pbar:
                for result in pool.imap(process_prompt, datas):
                    if result is not None:
                        results.append(result)
                        write_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        pbar.update(1)

    # 统计结果
    success = sum(1 for r in results if r['distill_status'] == 'success')
    fail = sum(1 for r in results if r['distill_status'] == 'fail')
    print(f"处理完成！成功: {success}, 失败: {fail}, 过滤: {len(datas)-success-fail}")

if __name__ == '__main__':
    main()