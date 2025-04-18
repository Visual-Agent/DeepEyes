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
    parser.add_argument('--output_dir', default='/cpfs/user/fengyuan/verl_data/r1-searcher/')

    args = parser.parse_args()

    sys_prompt = """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""
    sys_prompt = """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>

<answer> final answer here </answer>".
During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""

    stage1_data_fns = glob.glob(args.input_dir)
    stage1_data = []

    for fn in stage1_data_fns:
        with open(fn, 'r') as f:
            for line in f:
                line = json.loads(line)
                prompt = line['question']
                stage1_data.append({
                    "data_source": "rag_v2",
                    "prompt": [
                        {
                            "role": "system",
                            "content": sys_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }],
                    "question": line['question'],
                    "env_name": "rag_v2",
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
