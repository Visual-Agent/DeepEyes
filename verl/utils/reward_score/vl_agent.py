from openai import OpenAI
import requests
import random
import re
import os

from math_verify import parse, verify

openai_api_key = "EMPTY"
openai_api_base_list = [
    # "http://172.30.52.123:8000/v1",
    # "http://10.39.3.123:18901/v1",
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.39.3.123:18901/v1"),
]

client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
model_name_list = []
for client in client_list:
    response = requests.get(f"{api_base}/models")
    models = response.json()
    model_name_list.append(models['data'][0]['id'])



def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_dynamic_weights(history_stats: dict = None) -> dict:
    """
    基于历史工具调用统计计算动态权重
    
    参数:
        history_stats: 历史统计数据，包含：
            - avg_tool_calls: 平均工具调用次数
            - tool_success_rate: 工具调用成功率
            - tool_usage_rate: 工具使用率
            - history_size: 历史数据量
    
    返回:
        包含动态权重的字典：
            - acc_weight: 准确性奖励权重
            - format_weight: 格式奖励权重
            - tool_weight: 工具使用奖励权重
    """
    # 默认权重（与原固定权重一致）
    default_weights = {
        "acc_weight": 0.8,
        "format_weight": 0.2,
        "tool_weight": 1.2
    }
    
    if history_stats is None or history_stats.get("history_size", 0) == 0:
        return default_weights
    
    # 提取历史统计
    avg_tool_calls = history_stats.get("avg_tool_calls", 1.0)
    tool_success_rate = history_stats.get("tool_success_rate", 0.5)
    
    # 超参数配置
    expected_tool_calls = 1.5  # 期望的工具调用次数
    min_tool_calls = 0.3  # 最低期望的工具调用次数
    max_tool_calls = 3.0  # 最高期望的工具调用次数
    
    min_success_rate = 0.3  # 最低可接受的工具成功率
    target_success_rate = 0.7  # 理想的工具成功率
    
    # 1. 基于工具调用次数的调整因子
    # - 调用次数过少：鼓励工具使用 → 降低 acc 权重，提高 tool 权重
    # - 调用次数过多：可能过度依赖工具 → 提高 acc 权重，降低 tool 权重
    
    if avg_tool_calls < min_tool_calls:
        # 工具调用严重不足，需要大力鼓励
        tool_call_factor = 0.7  # 降低 acc 权重 30%
        tool_weight_factor = 1.5  # 提高 tool 权重 50%
    elif avg_tool_calls > max_tool_calls:
        # 工具调用过多，可能过度依赖
        tool_call_factor = 1.4  # 提高 acc 权重 40%
        tool_weight_factor = 0.6  # 降低 tool 权重 40%
    elif avg_tool_calls < expected_tool_calls * 0.8:
        # 工具调用略少，适度鼓励
        ratio = (avg_tool_calls - min_tool_calls) / (expected_tool_calls * 0.8 - min_tool_calls)
        tool_call_factor = 0.7 + 0.3 * ratio  # 从 0.7 线性增加到 1.0
        tool_weight_factor = 1.5 - 0.5 * ratio  # 从 1.5 线性减少到 1.0
    elif avg_tool_calls > expected_tool_calls * 1.5:
        # 工具调用略多，适度抑制
        ratio = (avg_tool_calls - expected_tool_calls * 1.5) / (max_tool_calls - expected_tool_calls * 1.5)
        tool_call_factor = 1.0 + 0.4 * ratio  # 从 1.0 线性增加到 1.4
        tool_weight_factor = 1.0 - 0.4 * ratio  # 从 1.0 线性减少到 0.6
    else:
        # 工具调用次数在合理范围内
        tool_call_factor = 1.0
        tool_weight_factor = 1.0
    
    # 2. 基于工具成功率的调整因子
    # - 成功率低：工具使用无效 → 提高 acc 权重，降低 tool 权重
    # - 成功率高：工具使用有效 → 保持或略微提高 tool 权重
    
    if tool_success_rate < min_success_rate:
        # 工具成功率极低，工具使用基本无效
        success_factor = 1.6  # 大幅提高 acc 权重
        success_tool_factor = 0.3  # 大幅降低 tool 权重
    elif tool_success_rate > target_success_rate:
        # 工具成功率很高，工具使用有效
        success_factor = 0.9  # 略微降低 acc 权重
        success_tool_factor = 1.2  # 略微提高 tool 权重
    else:
        # 成功率在中间范围，线性插值
        ratio = (tool_success_rate - min_success_rate) / (target_success_rate - min_success_rate)
        success_factor = 1.6 - 0.7 * ratio  # 从 1.6 线性减少到 0.9
        success_tool_factor = 0.3 + 0.9 * ratio  # 从 0.3 线性增加到 1.2
    
    # 3. 计算最终动态权重
    base_acc = default_weights["acc_weight"]
    base_format = default_weights["format_weight"]
    base_tool = default_weights["tool_weight"]
    
    # 组合调整因子
    final_acc_factor = tool_call_factor * success_factor
    final_tool_factor = tool_weight_factor * success_tool_factor
    
    # 计算最终权重，并限制在合理范围内
    dynamic_acc_weight = base_acc * final_acc_factor
    dynamic_acc_weight = max(0.3, min(1.8, dynamic_acc_weight))  # 限制范围 [0.3, 1.8]
    
    # 格式权重保持相对稳定，但可以根据 acc 权重进行适度调整
    # 确保格式惩罚不会被过度放大或缩小
    dynamic_format_weight = base_format * (1.0 + 0.3 * (1.0 - final_acc_factor))
    dynamic_format_weight = max(0.1, min(0.4, dynamic_format_weight))  # 限制范围 [0.1, 0.4]
    
    dynamic_tool_weight = base_tool * final_tool_factor
    dynamic_tool_weight = max(0.2, min(2.0, dynamic_tool_weight))  # 限制范围 [0.2, 2.0]
    
    return {
        "acc_weight": dynamic_acc_weight,
        "format_weight": dynamic_format_weight,
        "tool_weight": dynamic_tool_weight,
        "base_acc_weight": base_acc,
        "base_tool_weight": base_tool,
        "tool_call_factor": tool_call_factor,
        "success_factor": success_factor,
        "avg_tool_calls": avg_tool_calls,
        "tool_success_rate": tool_success_rate
    }


def compute_score(predict_str: str, ground_truth: str, extra_info=None, history_stats=None) -> float:
    """
    计算视觉语言任务的奖励，支持基于历史统计的动态权重
    
    参数:
        predict_str: 模型生成的响应字符串
        ground_truth: 标准答案
        extra_info: 额外信息（如问题文本）
        history_stats: 历史统计数据，用于计算动态权重
    """
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    question_text = extra_info['question']
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    # print(response)
    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    # 计算各奖励分量
    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    
    # 获取动态权重
    weights = compute_dynamic_weights(history_stats)
    
    # 使用动态权重计算最终奖励
    final_reward = (
        weights["acc_weight"] * acc_reward + 
        weights["format_weight"] * format_reward + 
        weights["tool_weight"] * tool_reward
    )
    
    # 打印调试信息（显示权重调整情况）
    if history_stats and history_stats.get("history_size", 0) > 0:
        print(f" [Dynamic Weights] acc={weights['acc_weight']:.2f} (base={weights['base_acc_weight']:.2f}), "
              f"tool={weights['tool_weight']:.2f} (base={weights['base_tool_weight']:.2f}), "
              f"history_calls={weights['avg_tool_calls']:.2f}, "
              f"history_success={weights['tool_success_rate']:.2f}")
    
    # 返回字典形式，包含详细信息以便追踪
    return {
        "score": final_reward,
        "acc": acc_reward,
        "format": format_reward,
        "tool": tool_reward,
        "tool_calls": count_vision_1,
        "acc_weight": weights["acc_weight"],
        "format_weight": weights["format_weight"],
        "tool_weight": weights["tool_weight"],
        "base_acc_weight": weights["base_acc_weight"],
        "base_tool_weight": weights["base_tool_weight"]
    }



def compute_common_reasoning(predict_str: str, ground_truth: str, extra_info=None, history_stats=None) -> float:
    """
    计算通用推理任务的奖励，支持基于历史统计的动态权重
    
    参数:
        predict_str: 模型生成的响应字符串
        ground_truth: 标准答案
        extra_info: 额外信息（如问题文本）
        history_stats: 历史统计数据，用于计算动态权重
    """
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = extract_answer(predict_no_think) # predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    if not answer_text:
        acc_reward = 0.0
        is_format_error = True
    elif len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    else:
        question_text = extra_info['question']
        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]
        full_prompt = COMMON_VERIFY_PROMPT.format(
            query=question_text,
            gold_ans=ground_truth,
            pred_ans=answer_text,
        )

        acc_reward = 0.0
        for ix in range(8):
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            judgement = response.split('## Equivalence Judgement')[-1].lower()
            if 'true' in judgement and 'false' not in judgement:
                acc_reward = 1.0
                break
            elif 'false' in judgement and 'true' not in judgement:
                acc_reward = 0.0
                break
            else:
                print(f' [ERROR] judgement format invalid: {judgement}')
                continue

    # 计算各奖励分量
    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    
    # 获取动态权重
    weights = compute_dynamic_weights(history_stats)
    
    # 使用动态权重计算最终奖励
    final_reward = (
        weights["acc_weight"] * acc_reward + 
        weights["format_weight"] * format_reward + 
        weights["tool_weight"] * tool_reward
    )
    
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {answer_text=}, {acc_reward=}, {format_reward=}')
    print(f' [Dynamic Weights] acc={weights["acc_weight"]:.2f}, tool={weights["tool_weight"]:.2f}')
    
    # 返回字典形式，包含详细信息以便追踪
    return {
        "score": final_reward,
        "acc": acc_reward,
        "format": format_reward,
        "tool": tool_reward,
        "tool_calls": count_vision_1,
        "acc_weight": weights["acc_weight"],
        "format_weight": weights["format_weight"],
        "tool_weight": weights["tool_weight"]
    }


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        print(f' [ERROR math] verify bug output: ')


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None, history_stats=None) -> float:
    """
    计算数学视觉推理任务的奖励，支持基于历史统计的动态权重
    
    参数:
        predict_str: 模型生成的响应字符串
        ground_truth: 标准答案
        extra_info: 额外信息（如问题文本）
        history_stats: 历史统计数据，用于计算动态权重
    """
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split('</think>')[-1].strip()
    answer_pattern = r'\\boxed{([^}]+)}'
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(extra_info['question'], ground_truth, model_answer) else 0.0
    
    format_reward = -1.0 if is_format_error else 0.0
    
    # 数学任务通常不涉及工具调用，但为了接口一致性，我们仍然支持动态权重
    # 这里使用不同的基础权重（数学任务更强调准确性）
    if history_stats is None or history_stats.get("history_size", 0) == 0:
        # 没有历史数据，使用原始固定权重
        final_reward = 1.2 * acc_reward + 0.4 * format_reward
        acc_weight = 1.2
        format_weight = 0.4
    else:
        # 有历史数据，使用动态权重
        # 数学任务的基础权重与视觉任务不同
        base_acc = 1.2
        base_format = 0.4
        
        # 基于历史统计进行调整
        # 对于数学任务，我们主要关注格式正确性和答案准确性
        # 如果历史格式错误率高，增加格式权重
        # 这里我们简化处理，直接使用基础权重，但可以根据需要扩展
        
        final_reward = base_acc * acc_reward + base_format * format_reward
        acc_weight = base_acc
        format_weight = base_format
    
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    print(f' [Weights] acc={acc_weight:.2f}, format={format_weight:.2f}')
    
    # 返回字典形式，包含详细信息以便追踪
    return {
        "score": final_reward,
        "acc": acc_reward,
        "format": format_reward,
        "acc_weight": acc_weight,
        "format_weight": format_weight
    }


if __name__ == '__main__':
    predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {'answer': 'The woman is to the left of the man who is holding the camera.', 'id': 0, 'image': '/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg', 'pred_ans': 'The woman is to the right of the man who is holding the camera.', 'question': 'Is the woman to the left or to the right of the man who is holding the camera?'}

    score = compute_score(predict_str, ground_truth, extra_info)
    print(f"Score: {score}")