import requests
import json
import argparse
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
import time, re, sys
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.workers.agent.envs.collab_code.sandbox_verify import RunCodeRequest, RunStatus, run_code_in_sandbox, OJConfig, oj_in_sandbox, OJRequest
from verl.workers.agent.envs.collab_code.system_prompt import USER_SP, ASSISTANT_SP, CODE_INITIAL_PROMPT



# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""

PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n"

N_TESTSET_PER_DATASET = 512  # per dataset
_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}


DIVIDE_SP_1 = """You are given a complex and information-rich instruction. Your task is to decompose this instruction into:
1. One **core instruction** that captures the essence and overall goal of the original instruction.  
2. Several **supplementary instructions** that elaborate on or add details to the core instruction. These supplementary instructions must be strongly related to and dependent on the core instruction — they should not function as independent tasks.

The goal is **not** to split the complex instruction into multiple independent tasks, but to extract a structured representation where **one core instruction summarizes the main task**, and each supplementary instruction provides additional, context-specific clarifications, constraints, or elaborations that support the core instruction.

Ensure that:  
- The **core instruction** stands as a concise and accurate abstraction of the full intent.  
- Each **supplementary instruction** refines or extends the core instruction without diverging into a separate goal.

Use clear and precise language. Do not omit any key information from the original instruction — ensure all content is preserved through the combination of the core and supplementary instructions.

**Example Input**:
"Write a Python script that reads a CSV file containing user information, filters out users under the age of 18, sorts the remaining users by their registration date in descending order, and then saves the result to a new CSV file with UTF-8 encoding."

**Example Output**:
Core Instruction: Write a Python script to process user data from a CSV file.

Supplementary Instructions:
1. Filter out users who are under the age of 18.
2. Sort the remaining users by registration date in descending order.
3. Save the processed data to a new CSV file using UTF-8 encoding.

**Output Format** (strictly follow this format, DO NOT PRINT ANYTHING ELSE!):

{{
    "Core Instruction": "<insert core instruction here>",
    "Supplementary Instructions": ["<first supplementary instruction>", "<second supplementary instruction>", ...]
}}

Now process this instruction as input:
{prompt}
"""

DIVIDE_SP_2 = """You are given a complex, information-rich instruction describing a search intent.  
Your task is to extract and restructure this intent into two parts:  

1. One **core instruction** — this should distill the **central, high-level search concept**, possibly vague or abstract, that best represents the overall goal.  
2. Several **supplementary instructions** — these must **add specificity** by introducing supporting search details, such as filters, constraints, desired attributes, or context. These should always depend on and refine the core instruction; they must not become standalone or unrelated search tasks.

Your goal is **not** to split the search into separate or independent queries, but to derive a structured representation of the search intent where:
- The **core instruction** reflects the primary topic or target of the search.  
- Each **supplementary instruction** contributes additional details that sharpen or contextualize that search.

Requirements:  
- The **core instruction** should be concise, generalizable, and clearly express the key idea.  
- Each **supplementary instruction** must clarify, constrain, or enrich the core instruction.  
- Preserve all original information — the combination of core and supplementary instructions must fully cover the original input.

Use precise and natural language appropriate for a search assistant or intelligent query processor.

**Example Input**:  
"Find cozy cafes in Tokyo that have Wi-Fi, are open late at night, and are good for reading alone."  

**Example Output**:  
Core Instruction: Search for cozy cafes in Tokyo.  

Supplementary Instructions:  
1. The cafe should offer Wi-Fi.  
2. It should be open late at night.  
3. It should be suitable for reading alone.

**Output Format** (strictly follow this format, DO NOT PRINT ANYTHING ELSE!):

{{
    "Core Instruction": "<insert core instruction here>",
    "Supplementary Instructions": ["<first supplementary instruction>", "<second supplementary instruction>", ...]
}}

Now process this instruction as input:
{prompt}
"""

DIVIDE_SP = {"v1": DIVIDE_SP_1, "v2": DIVIDE_SP_2}
version = "v1"


def get_claude_completion(prompt):
    """调用Claude API获取结果"""
    url = 'https://runway.devops.rednote.life/openai/bedrock_runtime/model/invoke'
    headers = {
        'token': 'd32974a62c3d471da311064046166621',
        'Content-Type': 'application/json'
    }
    data = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()['content'][0]['text']
    except Exception as e:
        logging.error(f"API请求失败: {str(e)}")
        raise


def extract_code_from_string(solution_str):
    CODE_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return '\n'.join(code_blocks).strip()


def code_eval(solution_code):
    result = run_code_in_sandbox(RunCodeRequest(code=solution_code, language='python', run_timeout=30), connection_timeout=120)
    succ = (result.status == RunStatus.Success)
    output = result.run_result.stderr
    return succ, output


def code_eval_stdio(solution_code, stdin, stdout):
    unitests = [{'input': {'stdin': inp}, 'output': {'stdout': oup}} for inp, oup in zip(stdin, stdout)]
    oj_data = {
        "id": 1,                          # Unique identifier
        "content": '',                     # Problem statement
        "test": unitests
    }
    result = oj_in_sandbox(OJRequest(
        completion=solution_code,
        config=OJConfig(language='python', provided_data=oj_data, extra={'run_all_cases': True}, run_timeout=30)),
                        connection_timeout=120)
    return result.accepted, ''


def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


def call_claude(prompt):
    max_retries = 100
    retry_delay = 3
    
    divide_result = {}

    for attempt in range(max_retries):
        try:
            response = get_claude_completion(DIVIDE_SP[version].format(prompt=prompt))
            response_dict = json.loads(response)
            divide_result["core"] = response_dict["Core Instruction"].strip()
            divide_result["supp"] = [ins.strip() for ins in response_dict["Supplementary Instructions"]]
            return divide_result
        except Exception as e:
            # print(response)
            if attempt < max_retries - 1:
                print(e)
                logging.warning(f"第{attempt+1}次重试: {prompt[:50]}...")
                time.sleep(retry_delay)
                continue
            return None


def taco():
    dataset = load_dataset("likaixin/TACO-verified")["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            oracle = json.loads(example["input_output"])
            source = example["source"]

            # skip poorly formatted examples
            if source in ["geeksforgeeks", "leetcode"]:
                return _EMPTY_RETURN_

            # too short description
            if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
                return _EMPTY_RETURN_

            # no image
            if "image" in example["question"].lower() or "\n![" in example["question"]:
                return _EMPTY_RETURN_
            
            if example["starter_code"].strip():
                return _EMPTY_RETURN_

            question = example["question"].strip()
            
            divide_result = call_claude(question)

            # prompt_pieces = [
            #     "Solve the programming task below in a Python markdown code block.",
            #     example["question"].strip(),
            # ]
            
            if "fn_name" in oracle:
                return _EMPTY_RETURN_
            
            elif "inputs" in oracle and "outputs" in oracle:
                stdin_list, stdout_list = minimize_stdio(oracle["inputs"], oracle["outputs"])
                if len(stdin_list) == 0:
                    return _EMPTY_RETURN_

                with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
                    futures = []
                    for stdin, stdout in zip(stdin_list, stdout_list):
                        futures.append(executor.submit(
                            code_eval_stdio,
                            example["solutions"][-1],
                            stdin,
                            stdout,
                        ))
                    for future in as_completed(futures):
                        pass_test, _ = future.result()
                        if not pass_test:
                            print('stdio-like test failed.')
                            return _EMPTY_RETURN_

                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown ground truth format: {oracle}")

            prompt = "\n".join(prompt_pieces)
            return {
                "data_source": "code",
                "prompt": [
                    {
                        "role": "system",
                        "content": ASSISTANT_SP["code"]
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (example["solutions"][0] if example["solutions"] else ""),
                    "dataset": "likaixin/TACO-verified",
                },
            }

        return process_fn

    dataset = dataset.map(function=make_map_fn("train"),
                          with_indices=True,
                          num_proc=64,
                          remove_columns=dataset.column_names).filter(lambda x: x != _EMPTY_RETURN_)
    splits = dataset.train_test_split(test_size=max(1, min(N_TESTSET_PER_DATASET, len(dataset) * 0.1)), seed=666)
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    for t in dataset:
        print(f"{t = }")
        t["extra_info"]["split"] = "test"

    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="./data/taco/train-00000-of-00001.parquet")
    parser.add_argument('--output', type=str, default="./data/taco/split_prompt/train-claude.jsonl")
    parser.add_argument('--processes', type=int, default=cpu_count(), 
                       help=f'并发进程数（默认: {cpu_count()}）')
    args = parser.parse_args()

    # 读取输入文件
    print(f"正在读取文件: {args.input}")
    # datas = read_jsonl(args.input)
    df = pd.read_parquet(args.input)
    datas = df.to_dict(orient='records')
    print(f"成功读取 {len(datas)} 条data")

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
    success = sum(1 for r in results if r['divide_status'] == 'success')
    logging.info(f"处理完成！成功: {success}, 失败: {len(datas)-success}")

if __name__ == '__main__':
    main()