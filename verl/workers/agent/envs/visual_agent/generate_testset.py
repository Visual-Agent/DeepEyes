import os
import datasets
import json
import glob
import argparse
import random
from tqdm import tqdm
random.seed(42)

def load_data_from_jsonl(target_fn):
    result_list = []
    with open(target_fn, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [ln.strip() for ln in lines if ln.strip() != ''] 
    for ln in lines:
        obj = json.loads(ln)
        obj['src_file'] = target_fn
        result_list.append(obj)
    # print(f' [*] load {len(result_list)} from {target_fn}')
    return result_list


def write_to_jsonl(output_filename, samples):
    with open(output_filename, 'w', encoding='utf-8') as f:
        for s in tqdm(samples, desc=f'writing to {output_filename}'):
            stext = json.dumps(s, ensure_ascii=False)
            f.write(stext)
            f.write('\n')
    print(f' [*] write {len(samples)} samples to {output_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/cpfs/user/fengyuan/code/github/R1-Searcher/data/eval_set/*.jsonl')
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

    stage1_data_fns = glob.glob(args.input_dir)
    stage1_data = []

    for fn in stage1_data_fns:
        with open(fn, 'r') as f:
            for line in f:
                line = json.loads(line)
                prompt = line['question']
                all_prompt = f"{prompt}\n\nPlease reason step by step, and put your final answer within <answer></answer>."

                stage1_data.append({
                    "data_source": "rag_v2-test",
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
                        # "id": line['idx'],
                        "question": prompt,
                        'answer': line['answer'],
                        # "pred_anses": line['pred_anses']
                    }
                })

    test_dataset = datasets.Dataset.from_list(stage1_data)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_dataset.to_parquet(os.path.join(args.output_dir, 'test.parquet'))
