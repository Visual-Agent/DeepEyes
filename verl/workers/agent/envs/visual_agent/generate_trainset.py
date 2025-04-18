import os
import datasets
import json
import argparse
import random
random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_data_path', default='/cpfs/user/honglingyi/DATA/LLM/RAG-RL-Hotpotqa-with-2wiki/stage_1.jsonl')
    parser.add_argument('--stage2_data_path', default='/cpfs/user/honglingyi/DATA/LLM/RAG-RL-Hotpotqa-with-2wiki/stage_2.jsonl')
    parser.add_argument('--output_dir', default='/cpfs/user/fengyuan/verl_data/mmsearch/')

    args = parser.parse_args()

    sys_prompt = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "text_search", "description": "text search using search engine", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", description": "Keywords for query."}, "max_results": {"type": "integer", "description": "Max number of search results. Value should be within the range 1-10. Defaults to 5."}}, "required": ["keyword"]}}}
{"type": "function", "function": {"name": "news_search", "description": "search for latest news using search engine", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", description": "Keywords for query."}, "max_results": {"type": "integer", "description": "Max number of search results. Value should be within the range 1-10. Defaults to 5."}}, "required": ["keyword"]}}}
{"type": "function", "function": {"name": "browse", "description": "capture a website screenshot with a given url", "parameters": {"type": "object", "properties": {"url": {"type": "string", description": "the website url to be captured"}}, "required": ["keyword"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

    prompt_set = set()
    num_filtered = 0

    stage1_data_path = args.stage1_data_path
    stage1_data = []
    with open(stage1_data_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            prompt = line['question']

            if prompt in prompt_set:
                num_filtered += 1
                # print(f' [DEBUG] filter duplicated {prompt=}')
                continue
            prompt_set.add(prompt)

            all_prompt = f"{prompt}\n\nPlease reason step by step, and put your final answer within <answer></answer>."
            stage1_data.append({
                "data_source": "rag_v2-train",
                "prompt": [
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": all_prompt,
                    }],
                "question": line['question'],
                "env_name": "mm_search",
                "ability": "qa",
                "reward_model": {
                        "style": "rule",
                        "ground_truth": line['answer']
                    },
                "extra_info": {
                    "id": line['idx'],
                    "question": prompt,
                    'answer': line['answer'],
                    # "pred_anses": line['pred_anses']
                }
            })

    stage2_data_path = args.stage2_data_path
    stage2_data = []
    with open(stage2_data_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            prompt = line['question']

            if prompt in prompt_set:
                num_filtered += 1
                # print(f' [DEBUG] filter duplicated {prompt=}')
                continue
            prompt_set.add(prompt)

            all_prompt = f"{prompt}\n\nPlease reason step by step, and put your final answer within <answer></answer>."
            stage2_data.append({
                "data_source": "rag_v2-train",
                "prompt": [
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": all_prompt,
                    }],
                "question": line['question'],
                "ability": "qa",
                "env_name": "mm_search",
                "reward_model": {
                        "style": "rule",
                        "ground_truth": line['answer']
                    },
                "extra_info": {
                    "id": line['idx'],
                    'answer': line['answer'],
                    "question": prompt
                }
            })

    print(f' [DEBUG] unique_prompt={len(prompt_set)}, {num_filtered=}')

    stage1_dataset = datasets.Dataset.from_list(stage1_data)
    stage2_dataset = datasets.Dataset.from_list(stage2_data)
    total_dataset = datasets.concatenate_datasets([stage1_dataset, stage2_dataset])
    total_dataset = total_dataset.shuffle()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    stage1_dataset.to_parquet(os.path.join(args.output_dir, 'stage1.parquet'))
    stage2_dataset.to_parquet(os.path.join(args.output_dir, 'stage2.parquet'))
    total_dataset.to_parquet(os.path.join(args.output_dir, 'train.parquet'))
