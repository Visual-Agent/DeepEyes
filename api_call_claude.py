import requests
import json
import argparse
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

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
    "Core Instruction": <insert core instruction here>,
    "Supplementary Instructions": [<first supplementary instruction>, <second supplementary instruction>, ...]
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
    "Core Instruction": <insert core instruction here>,
    "Supplementary Instructions": [<first supplementary instruction>, <second supplementary instruction>, ...]
}}

Now process this instruction as input:
{prompt}
"""

DIVIDE_SP = {"v1": DIVIDE_SP_1, "v2": DIVIDE_SP_2}
version = "v2"


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

def process_prompt(data):
    """处理单个prompt（包含重试机制）"""
    max_retries = 100
    retry_delay = 3
    
    # prompt = data['meta']['question_title'].strip()
    prompt = data['text'].strip()
    save_data = data
    
    for attempt in range(max_retries):
        try:
            response = get_claude_completion(DIVIDE_SP[version].format(prompt=prompt))
            try:
                response_dict = json.loads(response)
            except:
                print(response)
            save_data["collab_divide"] = {
                "Core Instruction": response_dict["Core Instruction"].strip(),
                "Supplementary Instructions": [ins.strip() for ins in response_dict["Supplementary Instructions"]],
                "divide_sp": version,
                "divide_model": "claude-3.7"
            }
            save_data["divide_status"] = "success"
            return save_data
        except Exception as e:
            if attempt < max_retries - 1:
                print(e)
                logging.warning(f"第{attempt+1}次重试: {prompt[:50]}...")
                time.sleep(retry_delay)
                continue
            save_data["divide_status"] = "error"
            return save_data

def read_jsonl(file_path):
    """读取JSONL文件并提取prompt字段"""
    datas = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                datas.append(data)
            except json.JSONDecodeError:
                logging.error(f"第{line_num}行JSON解析失败")
    return datas

def main():
    parser = argparse.ArgumentParser(description='Claude数据蒸馏工具')
    parser.add_argument('--input', type=str, help='输入JSONL文件路径', default="./data/collab_search/queries.jsonl")
    parser.add_argument('--output', type=str, help='输出JSONL文件路径', default="./data/collab_search/queries-claude.jsonl")
    parser.add_argument('--processes', type=int, default=cpu_count(), 
                       help=f'并发进程数（默认: {cpu_count()}）')
    args = parser.parse_args()

    # 读取输入文件
    print(f"正在读取文件: {args.input}")
    datas = read_jsonl(args.input)
    print(f"成功读取 {len(datas)} 条data")

    # 创建进程池
    with open(args.output, 'a+') as write_f:
        with Pool(processes=args.processes) as pool:
            results = []
            with tqdm(total=len(datas), desc="处理进度") as pbar:
                for result in pool.imap(process_prompt, datas):
                    results.append(result)
                    write_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    pbar.update(1)

    # 统计结果
    success = sum(1 for r in results if r['divide_status'] == 'success')
    logging.info(f"处理完成！成功: {success}, 失败: {len(datas)-success}")

if __name__ == '__main__':
    main()