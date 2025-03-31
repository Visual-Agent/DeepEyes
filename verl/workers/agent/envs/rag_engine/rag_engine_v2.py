import re
import random
import requests
import numpy as np
from time import sleep
from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents

class RAGEngineEnvV2(ToolBase):
    name = "rag_v2"

    valid_url_list = [
        "http://10.39.5.6:25111/queries",
        "http://10.39.5.6:25112/queries",
        "http://10.39.5.6:25113/queries",
        "http://10.39.5.6:25114/queries",
        "http://10.39.5.6:25115/queries",
        "http://10.39.5.6:25116/queries",
        "http://10.39.5.6:25117/queries",
        "http://10.39.5.6:25118/queries",
    ]

    topk = 3
    action_start = '<|begin_of_query|>'
    action_end = '<|end_of_query|>'
    answer_start = '<answer>'
    answer_end = '</answer>'
    doc_start = '<|begin_of_documents|>'
    doc_end = '<|end_of_documents|>'
    
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        answers = extract_tool_call_contents(self.answer_start, self.answer_end, action_string)
        if answers:
            # print(f' [DEBUG] found answer in {action_string=}')
            return '', 0.0, True, {}

        action_list = extract_tool_call_contents(self.action_start, self.action_end, action_string)
        if not action_list:
            # print(f' [DEBUG] no action_list in {action_string=}')
            return '',  0.0, True, {}

        action_list = [action.strip() for action in action_list]
        search_results = self._batch_search(action_list)
        if 'answers' not in search_results or not search_results['answers']:
            print(f' [WARNING] {action_list=} has no search result : {action_list}')
            return 'SEARCH RESULT IS EMPTY', 0.0, False, search_results

        # assert len(action_list) == len(search_results['answers']), f'{action_list=}, {len(search_results["answers"])=}'
        doc_string = self._passages2string(search_results['queries'], search_results['answers'])
        docs_string = f"\n{self.doc_start}\n{doc_string}\n{self.doc_end}\n"
        return docs_string, 0.0, False, search_results

    def reset(self, *args, **kwargs):
        pass

    def _batch_search(self, queries, max_retry=32):
        payload = {
            "queries": queries,
            "k": self.topk,
        }

        for it in range(max_retry):
            try:
                target_url = random.choice(self.valid_url_list)
                resp = requests.post(target_url, json=payload)
                resjson = resp.json()
                assert 'queries' in resjson and 'answers' in resjson, f"Invalid {resjson=}"
                return resjson
            except Exception as err:
                print(f' [ERROR] err={err} -- retry for {it}')
                sleep(random.uniform(0.1, 1.0))
                continue
        return {}

    def _passages2string(self, search_keys, retrieval_result):
        format_reference = ''
        for key, results in zip(search_keys, retrieval_result):
            if len(results) == 0:
                format_reference += 'None\n'
                continue

            for idx, doc_item in enumerate(results):
                doc_item_clean = re.sub(r'^\d+\s+', '', doc_item).strip()
                # format_reference += f"Doc {idx+1}\nKeyword: {key}\nTitle: {title}\nContent: {text}"
                format_reference += f"({idx + 1}){doc_item_clean}\n"

        return format_reference.strip()
