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
from verl.workers.agent.envs.collab_code.sandbox_verify import RunCodeRequest, RunStatus, run_code_in_sandbox
from verl.workers.agent.envs.collab_code.system_prompt import *


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

def conv(client, conversations, meta, with_cot=False):
    completion = openai_complete(
        client=client,
        messages=conversations,
        api_base=meta['url'],
        api_key=meta['api_key'],
        model=meta['model'],
        n=1,
    )
    if with_cot:
        completion = '<think>\n' + completion
        content = match_think_tag(completion)
        reasoning_content = extract_think_content(completion)[0].strip()
        conversations.append({'role': 'assistant', 'content': content.strip()})
        return reasoning_content, content
    else:
        return completion

def extract_code_from_string(solution_str):
    CODE_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return '\n'.join(code_blocks).strip()

def code_eval(task, solution_str, additional_import=True):
    # solution_code = extract_code_from_string(solution_str)
    if additional_import:
        solution_code = PY_IMPORTS + solution_str
    test_cases = f"{task['test']}\n\ncheck({task['entry_point'].strip()})"
    final_code = solution_code + "\n" + test_cases
    result = run_code_in_sandbox(RunCodeRequest(code=final_code, language='python', run_timeout=30), connection_timeout=120)
    exec_time = result.run_result.execution_time
    succ = (result.status == RunStatus.Success)
    print(succ)
    output = result.run_result.stderr
    return succ, output

# def code_gen_format(text):
#     pattern = r'(### Question:)(.*?)(### Format:)'
#     replaced_text = re.sub(pattern, r'\1\n{}\n\n\n\n\3', text, flags=re.DOTALL)
#     return replaced_text
def extract_code_content(content):
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    return matches

def match_think_tag(content):
    pattern = r'<think>.*?</think>'
    matches = re.sub(pattern, '', content, flags=re.DOTALL)
    return matches

def extract_think_content(content):
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    return matches

def extract_answer_content(content):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def generation(task, meta_dict, sp_dict, with_cot=False):
    logging.info(f"处理任务ID: {task.get('id', 'unknown')}")
    
    assistant_client = OpenAI(api_key=meta_dict['assistant']['api_key'], base_url=meta_dict['assistant']['url'])
    user_client = OpenAI(api_key=meta_dict['user']['api_key'], base_url=meta_dict['user']['url'])

    assistant_convs = [{'role': 'system', 'content': sp_dict['assistant']}]
    core = task['collab_divide']['Core Instruction']
    hidden = '\n'.join([f'{i+1}: {supp}' for i, supp in enumerate(task['collab_divide']['Supplementary Instructions'])])
    user_convs = [{'role': 'system', 'content': sp_dict['user'].format(core=core, hidden=hidden)}]
    
    starter_code = extract_code_content(task['query'])[0].strip()
    user_prompt = CODE_INITIAL_PROMPT.format(question=core, code=starter_code)
    finish = False
    turns = 0
    final_answer = ''
    
    # user_convs.append({'role': 'user', 'content': ''})
    # user_convs.append({'role': 'assistant', 'content': user_prompt})
    
    result = {'core': core, 'hidden': hidden}
    
    while turns < 10:
        assistant_convs.append({'role': 'user', 'content': user_prompt})
        response = conv(assistant_client, assistant_convs, meta_dict['assistant'], with_cot=with_cot)
        if with_cot:
            cot, response = response
        assistant_convs.append({'role': 'assistant', 'content': response})
        turns += 1
        
        # print('-'*20 + '\n' + f'TURN-{turns}' + '\n')
        # print('USER: ' + user_prompt + '\n' + '-'*20 + '\n')
        # print('ASSISTANT: ' + response + '\n' + '-'*20 + '\n')
        
        matches = extract_code_content(response)
        finish = (len(matches) != 0)
        
        if not finish:
            user_convs.append({'role': 'user', 'content': response})
            feedback = conv(user_client, user_convs, meta_dict['user'], with_cot=False)
            user_prompt = feedback
            user_convs.append({'role': 'assistant', 'content': user_prompt})
            # user_convs = [{'role': 'system', 'content': sp_dict['user'].format(hidden=hidden)}]  # reset, visible only current response to avoid cheating.
        else:
            final_answer = matches[0].strip()
            break
    
    # print('FINAL_ANSWER: ' + final_answer + '\n' + '-'*20 + '\n')
    
    result['ground_truth'] = task['completion']
    result['final_answer'] = final_answer
    if final_answer:
        result['test_success'], result['test_output'] = code_eval(task, final_answer)
    
    result['turns'] = turns
    result['trajectory'] = assistant_convs
    
    return result


def main(args):
    num_process = min(cpu_count(), 128)
    input_file = args.input_file
    output_file = args.output_file
    
    # setup_logging(output_file)
    
    logging.info(f'开始读取输入文件: {input_file}')
    with open(input_file, 'r') as f:
        # content = json.load(f)
        content = [json.loads(line) for line in f]
    logging.info(f'总共读取{len(content)}条数据')

    id_to_items = dict()
    for item in content:
        id_to_items[item['task_id']] = item
    
    tasks = []
    
    for id in id_to_items:
        item = id_to_items[id]
        tasks.append(item)
    logging.info(f'需要处理{len(tasks)}条数据')
    
    meta_dict = {
        'assistant': {
            'url': f'http://{args.assistant_url}:8000/v1',
            'model': args.assistant_model,
            'api_key': args.api_key,
            'temperature': 0.7,
            'max_tokens': 4096
        },
        'user': {
            'url': f'http://{args.user_url}:8000/v1',
            'model': args.user_model,
            'api_key': args.api_key,
            'temperature': 0.7,
            'max_tokens': 4096
        }
    }
    
    sp_dict = {
        'assistant': ASSISTANT_SP[args.task],
        'user': USER_SP[args.user]
    }

    with open(output_file, 'a+') as write_f:
        logging.info(f'启动{num_process}个进程处理任务')
        with ThreadPoolExecutor(max_workers=num_process) as executor:
            futures = []
            for sub_idx, task in enumerate(tasks):
                futures.append(executor.submit(generation, task, meta_dict, sp_dict))

            completed = 0
            failed = 0
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result is not None:
                    write_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    completed += 1
                    if completed % 100 == 0:
                        logging.info(f'已完成 {completed}/{len(tasks)} 个任务')
                else:
                    failed += 1
                    logging.warning(f'任务处理失败数: {failed}')
    
    logging.info(f'所有任务处理完成。成功: {completed}, 失败: {failed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default='./data/collab_code/split_prompt/LeetCodeDataset-v0.3.0-test-claude.jsonl')
    parser.add_argument('--output_file', type=str, default='./data/collab_code/qwen-7b-instruct/LeetCodeDataset-v0.3.0-test-auto_dialogue-low-test.jsonl')
    parser.add_argument('--assistant_model', type=str, default='qwen-7b-instruct')
    parser.add_argument('--assistant_url', type=str, default='10.39.21.158')
    parser.add_argument('--user_model', type=str, default='qwen-72b-instruct')
    parser.add_argument('--user_url', type=str, default='10.39.0.101')
    parser.add_argument('--api_key', type=str, default='zzw-114514')
    parser.add_argument('--user', type=str, default='low')
    parser.add_argument('--task', type=str, default='code')
    
    args = parser.parse_args()
    
    main(args)