from typing import Optional, List, Dict
import openai  # suggested version: 0.28
from openai import OpenAI
import random
import time
import csv
import re
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import os
import copy
import logging
from datetime import datetime
import argparse
from verl.workers.agent.envs.collab_code.code import extract_code_from_string
from verl.workers.agent.envs.collab_code.sandbox_verify import RunCodeRequest, RunStatus, run_code_in_sandbox


CODE_PROMPT = "Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\n{prompt}"

PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n"


def setup_logging(output_file):
    log_base = "/cpfs/user/zhengziwei/workspace/opencompass/outputs/judge_by_r1/Qwen-r1-32B-spec/logs"
    log_file = f"{log_base}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(log_base, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"开始运行采样任务，输出文件: {output_file}")

def openai_complete(
    client: openai.OpenAI,
    messages: List,
    system_prompt: Optional[str] = None,
    sleep_time: float = 100,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    try_times = 32,
    **kwargs,
):
    deployment_id = None
    openai.api_base = api_base
    openai.api_key = api_key
    retry = 0
    rate_error_flag = False
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            if 'error' in completion:
                logging.error(f"API返回错误: {completion['error']['message']}")
                raise openai.OpenAIError(completion['error']['message'])

            rate_error_flag = False
            if 'n' not in kwargs:
                choice = completion['choices'][0]
                assert choice['message']['role'] == "assistant"
                if 'content' in choice['message'] and choice['message']['content'] != "":
                    completion = choice['message']['content']
                else:
                    logging.error(f"无效的返回消息: {choice['message']}")
                    raise openai.OpenAIError(f"Invalid {choice['message']=}")
            rate_error_flag = False
            return completion.choices[0].message.content
        except (openai.OpenAIError, AttributeError) as e:
            if "Please reduce" in str(e) or "Detected an error in the prompt." in str(e):
                logging.warning(f"不重试的错误: \"{e}\"")
                return ""
            else:
                if not rate_error_flag:
                    logging.warning(f"遇到错误 \"{e}\", {sleep_time}秒后重试...")
                    rate_error_flag = True
                retry += 1
                if retry > try_times:
                    logging.error(f"达到最大重试次数{try_times}次，放弃该请求")
                    return ""
                time.sleep(sleep_time)

def extract_code_from_string(solution_str):
    CODE_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return '\n'.join(code_blocks).strip()

# def code_eval(task, solution_str, additional_import=True):
#     solution_code = extract_code_from_string(solution_str)
#     if additional_import:
#         solution_code = PY_IMPORTS + solution_code
#     test_cases = f"{task['test']}\n\ncheck({task['entry_point'].strip()})"
#     succ, output = code_exec(solution_code + "\n" + test_cases)
#     return succ, output

def code_eval(task, solution_str, additional_import=True):
    solution_code = extract_code_from_string(solution_str)
    if additional_import:
        solution_code = PY_IMPORTS + solution_code
    test_cases = f"{task['test']}\n\ncheck({task['entry_point'].strip()})"
    final_code = solution_code + "\n" + test_cases
    result = run_code_in_sandbox(RunCodeRequest(code=final_code, language='python', run_timeout=30), connection_timeout=120)
    exec_time = result.run_result.execution_time
    succ = (result.status == RunStatus.Success)
    output = result.run_result.stderr
    return succ, output

def code_gen_format(text):
    pattern = r"(### Question:)(.*?)(### Format:)"
    replaced_text = re.sub(pattern, r"\1\n{}\n\n\n\n\3", text, flags=re.DOTALL)
    return replaced_text

def generation(task, meta, prompt_template, cot=False):
    logging.info(f"处理任务ID: {task.get('id', 'unknown')}")
    client = OpenAI(api_key=meta['api_key'], base_url=meta['url'])

    kwargs = {
            # "max_tokens": meta.get("max_tokens", 8192),
            # "temperature": meta.get("temperature", 0.6),
    }
    
    conversations = [{"role": "system", "content": ""}]
    
    # question = task['meta']['question_title']
    # question = task['collab_divide']['Core Instruction']
    question = task['collab_divide']['Core Instruction'] + '\n' + '\n'.join(task['collab_divide']['Supplementary Instructions'])
    
    # if prompt_template:
    #     prompt = prompt_template.format(prompt=question)
    # else:
    #     prompt = question
    
    prompt = code_gen_format(task["query"]).format(question)
    
    conversations.append({"role": "user", "content": prompt})
    # result = task
    result = {"prompt": prompt}

    completion = openai_complete(
        client=client,
        messages=conversations,
        api_base=meta['url'],
        api_key=meta['api_key'],
        model=meta['model'],
        n=1,
        **kwargs,
    )
    if cot:
        completion = "<think>\n" + completion  # 无 ds system prompt
        content = match_think_tag(completion)
        reasoning_content = extract_think_content(completion)[0].strip()
        conversations.append({"role": "assistant", "content": content.strip()})
        result['cot'] = reasoning_content
        result['response'] = content
    else:
        result['response'] = completion
    
    result['test_success'], result['test_output'] = code_eval(task, result['response'])
   
    return result

def match_think_tag(content):
    # 使用正则表达式匹配并删除<think>标签之间的内容
    pattern = r'<think>.*?</think>'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    return content

def extract_think_content(content):
    # 使用正则表达式来匹配 <think> 标签之间的内容
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    return matches

def unwrap_sample_1_input(args):
    try:
        return generation(*args)
    except Exception as e:
        logging.error(f"处理任务时发生错误: {str(e)}")
        return None

def main(args):
    num_process = min(cpu_count(), 128)
    input_file = args.input_file
    output_file = args.output_file
    
    # setup_logging(output_file)
    
    logging.info(f"开始读取输入文件: {input_file}")
    with open(input_file, 'r') as f:
        # content = json.load(f)
        content = [json.loads(line) for line in f]
    logging.info(f"总共读取{len(content)}条数据")

    id_to_items = dict()
    for item in content:
        id_to_items[item['task_id']] = item
    
    tasks = []
    
    for id in id_to_items:
        item = id_to_items[id]
        tasks.append(item)
    logging.info(f"需要处理{len(tasks)}条数据")
    
    meta_list = []
    urls = [args.url]
    for url in urls:
        meta_list.append({
            "url": f"http://{url}:8000/v1",
            "model": args.model,
            "api_key": args.api_key,
            "temperature": 0.7,
            "max_tokens": 4096
        })
    
    prompt_template = CODE_PROMPT

    with open(output_file, 'a+') as write_f:
        logging.info(f"启动{num_process}个进程处理任务")
        with ThreadPoolExecutor(max_workers=num_process) as executor:
            futures = []
            for sub_idx, task in enumerate(tasks):
                futures.append(executor.submit(generation, task, meta_list[sub_idx % len(meta_list)], prompt_template))

            completed = 0
            failed = 0
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    write_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    completed += 1
                    if completed % 100 == 0:
                        logging.info(f"已完成 {completed}/{len(tasks)} 个任务")
                else:
                    failed += 1
                    logging.warning(f"任务处理失败数: {failed}")
    
    logging.info(f"所有任务处理完成。成功: {completed}, 失败: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default="./data/collab_code/LeetCodeDataset-v0.3.0-test-claude.jsonl")
    parser.add_argument('--output_file', type=str, default="./data/collab_code/qwen-72b-instruct/LeetCodeDataset-v0.3.0-test-pred-core+supp.jsonl")
    parser.add_argument('--model', type=str, default="qwen-72b-instruct")
    parser.add_argument('--api_key', type=str, default="zzw-114514")
    # parser.add_argument('--urls', type=json.loads, help='url lists of deployed models', default="["10.39.4.221"]")
    parser.add_argument('--url', type=str, default="10.39.0.61")
    
    args = parser.parse_args()
    
    main(args)