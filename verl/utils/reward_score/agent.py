# import re
# import evaluate
# import string
# from collections import Counter

# exact_match = evaluate.load("exact_match")

# def compute_score(data_source: str, predict_str: str, ground_truth: str) -> float:
#     if data_source.lower() == 'rag':
#         predict_str = '<think>' + predict_str
#         think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
#         think_match = re.search(think_pattern, predict_str)
#         if not think_match:
#             return 0.0

#         predict_no_think = predict_str.split('</think>')[-1]
#         answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
#         match_result = re.search(answer_pattern, predict_no_think)
#         if match_result:
#             answer_text = match_result.group(1).strip()
#             score_info = exact_match.compute(references=[ground_truth], predictions=[answer_text], ignore_case=True, ignore_punctuation=True)
#             acc_reward = float(score_info['exact_match'])
#             format_reward = 1.0
#         else:
#             acc_reward = 0.0
#             format_reward = 0.0
#         return 0.8 * acc_reward + 0.2 * format_reward

#     # For games the reward are given by env
#     return 0.0

import re
import evaluate
import string
from collections import Counter


exact_match = evaluate.load("exact_match")


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return f1, precision, recall


def compute_score(predict_str: str, ground_truth: str) -> float:
    is_format_error = False
    predict_str = '<think>' + predict_str
    count_1 = predict_str.count("<|begin_of_documents|>\n")
    count_2 = predict_str.count("<|end_of_documents|>\n")
    count_3 = predict_str.count("<|begin_of_query|>")
    count_4 = predict_str.count("<|end_of_query|>")
    count_5 = predict_str.count("<|begin_of_documents|>")
    count_6 = predict_str.count("<|end_of_documents|>")
    count_7 = predict_str.count("<|begin_of_documents|>\n(1)")
    if not (count_1 == count_2 == count_3 == count_4 == count_5 == count_6 == count_7):
        is_format_error = True

    count_assiatant_1 = predict_str.count("Assistant:")
    count_assiatant_2 = predict_str.count("assistant:")
    if count_assiatant_1 != 0 or count_assiatant_2 != 0:
        is_format_error = True

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != 1 or count_answer_2 != 1:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    if "begin_of_query" in answer_text or "begin_of_documents" in answer_text:
        is_format_error = True

    answer_len = len(answer_text.split())
    if answer_len > 10:
        is_format_error = True

    if count_7 == 0:
        is_format_error = True

    retrieval_pattern = re.compile(r'<\|begin_of_query\|>(.*?)<\|end_of_query\|>', re.DOTALL)
    retrieval_match = re.search(retrieval_pattern, predict_str)
    doc_pattern = re.compile(r'<\|begin_of_documents\|>(.*?)<\|end_of_documents\|>', re.DOTALL)
    doc_match = re.search(doc_pattern, predict_str)

    retrieval_reward = 1.0 if count_7 >= 1 else -1.0
    # em_score = exact_match.compute(references=[ground_truth], predictions=[answer_text], ignore_case=True, ignore_punctuation=True)
    acc_reward, _ , _ = f1_score(answer_text, ground_truth)
    acc_reward = 2.0 * acc_reward

    format_reward = -2.0 if is_format_error else 0.0
    return format_reward + retrieval_reward + acc_reward


def compute_score_eval(predict_str: str, ground_truth: str) -> float:
    predict_no_think = predict_str.split('</think>')[-1].strip()
    answer_text = predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    score_info = exact_match.compute(references=[ground_truth], predictions=[answer_text], ignore_case=True, ignore_punctuation=True)
    acc_reward = float(score_info['exact_match'])
    return acc_reward