import re
import random
import requests
import numpy as np
from typing import Optional, List, Dict
import openai
from openai import OpenAI
import time
from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents
from .system_prompt import USER_SP, PY_IMPORTS
from .sandbox_verify import RunCodeRequest, RunStatus, run_code_in_sandbox


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
                raise openai.OpenAIError(completion['error']['message'])

            rate_error_flag = False
            if 'n' not in kwargs:
                choice = completion['choices'][0]
                assert choice['message']['role'] == "assistant"
                if 'content' in choice['message'] and choice['message']['content'] != "":
                    completion = choice['message']['content']
                else:
                    raise openai.OpenAIError(f"Invalid {choice['message']=}")
            rate_error_flag = False
            return completion.choices[0].message.content
        except (openai.OpenAIError, AttributeError) as e:
            if "Please reduce" in str(e) or "Detected an error in the prompt." in str(e):
                return ""
            else:
                if not rate_error_flag:
                    rate_error_flag = True
                retry += 1
                if retry > try_times:
                    return ""
                time.sleep(sleep_time)

def match_think_tag(content):
    pattern = r'<think>.*?</think>'
    matches = re.sub(pattern, '', content, flags=re.DOTALL)
    return matches

def extract_think_content(content):
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, content, flags=re.DOTALL)
    return matches


class UserFeedback(ToolBase):
    name = "collab_code"
    
    action_start = ''
    action_end = '<|im_end|>'
    code_start = '```python'
    code_end = '```'
    
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        # the whole response as an action to call the user model
        action_list = extract_tool_call_contents(self.action_start, self.action_end, action_string)
        if not action_list:
            return '',  0.0, True, {}
        current_response = action_list[0]
        
        # stop auto dialogue if found code solution in assistant's response
        answers = extract_tool_call_contents(self.code_start, self.code_end, current_response)
        if answers:
            solution_str = '\n'.join(answers)
            succ, stderr = self._code_eval(solution_str)
            reward = float(succ)
            return '', reward, True, {}
        
        user_feedback, _ = self._single_conv(current_response)
        self.convs.append({'role': 'assistant', 'content': user_feedback})
        
        obs = "<|im_start|>user\n" + user_feedback + "<|im_end|>\n" + "<|im_start|>assistant\n"
        
        return obs, 0.0, False, None

    def reset(self, *args, **kwargs):
        user_meta = kwargs.pop('user_meta')
        self.user_config = {
            'url': f'http://{user_meta.url}:8000/v1',
            'model': user_meta.name,
            'api_key': user_meta.api_key,
            'temperature': user_meta.temperature,
            'max_tokens': user_meta.max_tokens
        }
        self.with_cot = ('cot' in user_meta.name)
        user_initial_prompt = kwargs.pop('raw_prompt')[1]['content']
        core = user_initial_prompt.split('Starter code:')[0].strip()
        self.env_info = kwargs.pop('env_info')
        user_sp = USER_SP[user_meta.level]
        self.user_sp = user_sp.format(core=core, hidden=self.env_info['hidden'])
        self._create_convs()
    
    def _create_convs(self):
        self.convs = [{'role': 'system', 'content': self.user_sp}]
    
    def _single_conv(self, response):
        user_client = OpenAI(api_key=self.user_config['api_key'], base_url=self.user_config['url'])
        self.convs.append({'role': 'user', 'content': response})
        completion = openai_complete(
            client=user_client,
            messages=self.convs,
            api_base=self.user_config['url'],
            api_key=self.user_config['api_key'],
            model=self.user_config['model'],
            n=1,
        )
        if self.with_cot:
            completion = '<think>\n' + completion
            content = match_think_tag(completion)
            reasoning_content = extract_think_content(completion)[0].strip()
            return content, reasoning_content
        else:
            return completion, None
    
    def _code_eval(self, solution_str, additional_import=True):
        # solution_code = extract_code_from_string(solution_str)
        if additional_import:
            solution_code = PY_IMPORTS + solution_str
        test_cases = self.env_info['test_cases']
        final_code = solution_code + "\n" + test_cases
        try:
            result = run_code_in_sandbox(RunCodeRequest(code=final_code, language='python', run_timeout=30), connection_timeout=30)
            exec_time = result.run_result.execution_time
            succ = (result.status == RunStatus.Success)
            stderr = result.run_result.stderr
        except:
            succ = False
            stderr = 'sandox error'
            print(stderr)
        return succ, stderr
