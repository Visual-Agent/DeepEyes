import sys
import os
import logging
from enum import Enum
from typing import Dict, Literal, Optional
# import fire
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from pydantic import BaseModel
import requests
from tqdm import tqdm
from multiprocessing import Pool
import time
import json
import re
import ast
import random
from datetime import datetime
import pickle
from typing import Union
import string
import base64

DEFAULT_RUNTIMEOUT = 30
DEFAULT_CONNECTION_TIMEOUT = 120


MAX_RETRY=3

SUPPORT_LANGUAGES=['python', 'pytest', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'csharp', 'typescript', 'rust']

API_POOLS=['http://10.39.2.101:17795']

OJ_API_POOLS=['http://10.39.2.101:17710', 'http://10.39.2.101:17706']



cnt=0

def get_endpoint():
    API = random.choice(API_POOLS)
    # print(API)
    return f'{API}'

def get_oj_endpoint():
    API = random.choice(OJ_API_POOLS)
    # print(API)
    return f'{API}'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class CommandRunStatus(str, Enum):
    Finished = 'Finished'
    TimeLimitExceeded = 'TimeLimitExceeded'
    # ignore this in logic as this state cause run_code to throw
    Error = 'Error'


class CommandRunResult(BaseModel):
    status: CommandRunStatus
    execution_time: Optional[float] = None
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class RunCodeRequest(BaseModel):
    code: str
    language: Literal['python', 'pytest', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'csharp', 'typescript', 'rust']
    compile_timeout: float = 30
    run_timeout: float = DEFAULT_RUNTIMEOUT
    files: Dict[str, str] = {}

class OJConfig(BaseModel):
    language: Literal['python', 'pytest', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'csharp', 'typescript', 'rust']
    compile_timeout: float = 30
    run_timeout: float = DEFAULT_RUNTIMEOUT
    locale: str="en"
    dataset_type: str="CommonOJDataset"
    provided_data: dict = {}
    extra: dict = {}

class OJRequest(BaseModel):
    completion: str
    dataset: str='code_contests_train'
    id: str=''
    config: OJConfig

class RunStatus(str, Enum):
    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side, ignore this in logic as this state cause run_code to throw
    SandboxError = 'SandboxError'

class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None


class OJResponse(BaseModel):
    extracted_code: str
    accepted: bool = None
    tests: list = None

def on_retry_error(s):
    e = s.outcome.exception()
    logger.error(f'give up requesting sandbox. error: {e}')
    raise e


def before_retry_sleep(s):
    logger.warning(f'error requesting sandbox for {s.attempt_number} time(s), will retry... error: {s.outcome.exception()}')


def is_malicuous_code(code: str) -> bool:
    if 'os.killpg' in code:
        return True
    else:
        return False

def save_failed_request(request_data: Union[RunCodeRequest, OJRequest]):
    # 生成文件名：时间戳 + 随机5字符（字母数字）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    filename = f"/cpfs/user/mjren/logs/sandboxfusion/aborted_{timestamp}_{random_str}.pkl"
    
    # 使用 pickle 存储对象
    with open(filename, 'wb') as f:  # 注意是二进制写入模式 'wb'
        pickle.dump(request_data, f)
    
    print(f"请求失败，已使用 pickle 存储到文件: {filename}")


# wait=wait_exponential_jitter(): 这是定义等待重试的时间策略。wait_exponential_jitter() 结合了指数递增和随机抖动的等待时间。每次重试之间的等待时间会成指数增长，但为了避免高峰或拥塞，它还会加上随机抖动。
# stop=stop_after_attempt(12): 这是定义重试的终止条件。这里表示最多会进行 12 次重试。
# before_sleep=before_retry_sleep: 这个参数指定了在每次重试前执行的操作，比如记录日志或处理其他逻辑。before_retry_sleep 可能是一个回调函数，用来在重试之前记录或处理某些数据。
# retry_error_callback=on_retry_error: 当所有的重试都失败后，会执行 on_retry_error 这个回调函数，用于处理最终的错误，可能记录日志或采取其他补救措施。
@retry(wait=wait_exponential_jitter(max=1),
       stop=stop_after_attempt(MAX_RETRY),
       before_sleep=before_retry_sleep,
       retry_error_callback=on_retry_error)
def run_code_in_sandbox(request: RunCodeRequest, connection_timeout) -> RunCodeResponse:
    # try:
    result = requests.post(get_endpoint()+'/run_code', 
                            json=request.model_dump(), 
                            timeout=connection_timeout)
    # except requests.exceptions.ConnectionError as e:
    #     save_failed_request(request)
    #     logger.warning(f'malicious code, server aborted')
    #     return None
    
    if result.status_code == 500 and ('error_code' in json.loads(result.text) and json.loads(result.text)['error_code'] == 'function_proxy_error'):
        return None
    if result.status_code != 200:
        raise Exception(f'Faas api responded with code {result.status_code}: {result.text}')
    resp = RunCodeResponse(**result.json())
    if resp.status == RunStatus.SandboxError:
        raise Exception(f'Sanbox responded with error: {resp.message}')
    logger.debug(f'sandbox request success. request = {request.model_dump_json(indent=2)}. response = {resp.model_dump_json(indent=2)}')
    return resp


@retry(wait=wait_exponential_jitter(max=1),
       stop=stop_after_attempt(MAX_RETRY),
       before_sleep=before_retry_sleep,
       retry_error_callback=on_retry_error)
def oj_in_sandbox(request: OJRequest, connection_timeout) -> RunCodeResponse:
    # try:
    result = requests.post(get_oj_endpoint()+'/submit', 
                            json=request.model_dump(), 
                            timeout=connection_timeout)
    # except requests.exceptions.ConnectionError as e:
    #     save_failed_request(request)
    #     logger.warning(f'malicious code, server aborted')
    #     return None
    
    if result.status_code == 500 and ('error_code' in json.loads(result.text) and json.loads(result.text)['error_code'] == 'function_proxy_error'):
        return None
    if result.status_code != 200:
        raise Exception(f'Faas api responded with code {result.status_code}: {result.text}')
    resp = OJResponse(**result.json())
    logger.debug(f'sandbox request success. request = {request.model_dump_json(indent=2)}. response = {resp.model_dump_json(indent=2)}')
    return resp


class SummaryMapping(BaseModel):
    Success: str = RunStatus.Success
    Failed: str = RunStatus.Failed
    CompileFailed: Optional[str] = None
    CompileTimeout: Optional[str] = None
    RunFailed: Optional[str] = None
    RunTimeout: Optional[str] = None


def summary_result(result: RunCodeResponse, mapping: SummaryMapping) -> str:
    if result.compile_result is None and result.run_result is None:
        # note: this should not happen
        if result.status == RunStatus.Success:
            return mapping.Success
        if result.status == RunStatus.Failed:
            return mapping.Failed
        raise Exception(f'unexpected result status {result.status}')
    if result.run_result is None:
        # compile error
        if result.compile_result.status == CommandRunStatus.TimeLimitExceeded:
            return mapping.CompileTimeout or mapping.Failed
        return_code = result.compile_result.return_code
        if return_code is None:
            raise Exception(f'invalid sandbox result: no return code with status {result.compile_result.status}')
        if return_code != 0:
            return mapping.CompileFailed or mapping.Failed
        raise Exception(f'invalid sandbox result: compiled succesfully with no run result')
    if result.run_result.status == CommandRunStatus.TimeLimitExceeded:
        return mapping.RunTimeout or mapping.Failed
    return_code = result.run_result.return_code
    # print(f'return_code {return_code}')


def extract_solution(solution_str, thinking, zero):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # TODO support extract from <answer> tag
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    
    # 正则匹配 <answer> 标签中可能存在的代码块 (包括语言和代码)
    # answer_pattern = r'<answer>.*?```(\w+)?\n(.*?)```.*?</answer>'
    # TODO为了支持多格式（部分模型的answer不会wrap），直接使用```提取
    if not thinking:
        answer_pattern = r'```(\w+)?\n(.*?)```'
        matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))  # re.DOTALL 允许匹配多行内容
    elif thinking and not zero:
        if '</think>' not in solution_str:
            return None, None
        else:
            answer_pattern = r'```(\w+)?\n(.*?)```'
            matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    elif thinking and zero:
        if '</think>' not in solution_str:
            return None, None
        else:
            answer_pattern = r'<answer>.*?```(\w+)?\n(.*?)```.*?</answer>'
            matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    # 如果找到匹配项，提取最后一个 <answer> 中的语言和代码
    if matches:
        last_match = matches[-1]
        language = last_match.group(1) or 'python'  # 如果没有指定语言，默认为 python
        code = last_match.group(2).strip()
        return language, code
    else:
        return None, None
    


def compute_score_stdio(solution_str, ground_truth, 
                        thinking, zero, import_str, 
                        run_timeout,
                        format_score, 
                        negative_format_score,
                        score,
                        connection_timeout):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    exec_time = None
    
    global cnt
    language, code = extract_solution(solution_str=solution_str, thinking=thinking, zero=zero)
    if language not in SUPPORT_LANGUAGES or code is None:
        return negative_format_score, exec_time
    if is_malicuous_code(code):
        return format_score, exec_time
    
    cnt += 1
    tests = ground_truth
    reward = 0
    oj_data = {
        "id": 1,                          # Unique identifier
        "content": '',                     # Problem statement
        "test": tests
    }
    try:
        result = oj_in_sandbox(OJRequest(completion=f"```{language}\n{import_str}\n\n{code}\n```", 
                                            config=OJConfig(language='python', provided_data=oj_data, 
                                                            extra={'run_all_cases': False},
                                                            run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT,)), 
                                connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT)
        exec_time = [(test['exec_info']['run_result']['execution_time'], test['test_info']) for test in result.tests]
        
        if result is None:
            reward = format_score
        else:
            # assert code == result.extracted_code, f'code not equal: {code} vs {result.extracted_code}'
            pass_rate = int(result.accepted)
            reward = pass_rate*score + format_score
    except requests.exceptions.ConnectionError as e:
        reward = format_score
    except Exception as e:
        logger.warning(f'oj_in_sandbox failed: {e}')
        reward = format_score
    return reward, exec_time
    
    
def compute_score_pytest(solution_str, pytest_code, 
                         thinking, zero, import_str, 
                         run_timeout,
                         format_score, 
                         negative_format_score,
                         score,
                         connection_timeout):
    exec_time = None
    global cnt
    language, code = extract_solution(solution_str=solution_str, thinking=thinking, zero=zero)
    if language not in SUPPORT_LANGUAGES or code is None:
        return negative_format_score, exec_time
    if is_malicuous_code(code):
        return format_score, exec_time
    
    cnt += 1
    reward = 0
    try:
        result = run_code_in_sandbox(RunCodeRequest(code=f'{import_str}\n\n{code}\n\n{pytest_code}', 
                                                    language='pytest',
                                                    run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT),            
                                        connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT)
        
        exec_time = result.run_result.execution_time
        if result is None:
            reward = format_score
        else:
            pass_rate = 1 if result.status == RunStatus.Success else 0
            reward = pass_rate*score+format_score
    except requests.exceptions.ConnectionError as e:
        reward = format_score
    except Exception as e:
        logger.warning(f'oj_in_sandbox failed: {e}')
        reward = format_score
    return reward, exec_time


def compute_score_assertions(solution_str, assertions, entry_point_str,
                             thinking, zero, import_str, 
                             run_timeout,
                             format_score, 
                             negative_format_score,
                             score,
                             connection_timeout):
    exec_time = None
    global cnt
    language, code = extract_solution(solution_str=solution_str, thinking=thinking, zero=zero)
    
    if language not in SUPPORT_LANGUAGES or code is None:
        return negative_format_score, exec_time
    if is_malicuous_code(code):
        return format_score, exec_time
        
    cnt += 1
    reward = 0
    solution=f"{import_str}\n\n{code}"
    encodeed_solution = base64.b64encode(solution.encode('utf-8')).decode('utf-8')
    try:
        assertions_code = "\n".join(assertions)
        
        if 'class Solution' not in solution:
            final_code = f"""
{import_str}
from solution import {entry_point_str}

{assertions_code}
            """
        else:
            final_code = f"""
{import_str}
from solution import Solution

{entry_point_str} = Solution().{entry_point_str}

{assertions_code}
            """
        
        result = run_code_in_sandbox(RunCodeRequest(code=final_code, 
                                                    language=language,
                                                    files={'solution.py': encodeed_solution},
                                                    run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT), 
                                        connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT)
        
        exec_time = result.run_result.execution_time
        if result is None:
            reward = format_score
        else:
            pass_rate = 1 if result.status == RunStatus.Success else 0
            reward = pass_rate*score+format_score
    except requests.exceptions.ConnectionError as e:
        reward = format_score
    except Exception as e:
        logger.warning(f'oj_in_sandbox failed: {e}')
        reward = format_score
        
    return reward, exec_time


def compute_score_call(solution_str, test_functions, entry_point_str,
                       thinking, zero, import_str, 
                       run_timeout,
                       format_score, 
                       negative_format_score,
                       score,
                       connection_timeout):
    exec_time = None
    global cnt
    language, code = extract_solution(solution_str=solution_str, thinking=thinking, zero=zero)
    
    if language not in SUPPORT_LANGUAGES or code is None:
        return negative_format_score, exec_time
    if is_malicuous_code(code):
        return format_score, exec_time
    
    cnt += 1
    reward = 0
    solution=f"{import_str}\n\n{code}"
    encodeed_solution = base64.b64encode(solution.encode('utf-8')).decode('utf-8')
    try:
        test_function_str = '\n\n'.join(test_functions)
        if '()' not in entry_point_str:
            final_code = f"""
{import_str}
from solution import {entry_point_str}

{test_function_str}

check({entry_point_str})
            """
        else:
            part1 = entry_point_str.split('().')[0]
            part2 = entry_point_str.split('().')[1]
            final_code = f"""
{import_str}
from solution import {part1}

{test_function_str}


check({part1}().{part2})
            """
        result = run_code_in_sandbox(RunCodeRequest(code=final_code, language=language,
                                                    files={'solution.py': encodeed_solution},
                                                    run_timeout=run_timeout if run_timeout else DEFAULT_RUNTIMEOUT), 
                                        connection_timeout=connection_timeout if connection_timeout else DEFAULT_CONNECTION_TIMEOUT)
        exec_time = result.run_result.execution_time
        if result is None:
            reward = format_score
        else:
            pass_rate = 1 if result.status == RunStatus.Success else 0
            if pass_rate == 0:
                pass
            reward = pass_rate*score+format_score
    except requests.exceptions.ConnectionError as e:
        reward = format_score
    except Exception as e:
        logger.warning(f'oj_in_sandbox failed: {e}')
        reward = format_score
    return reward, exec_time

