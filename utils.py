from __future__ import division
import torch
from torch.autograd import Function
import torch.nn as nn
from symbolics import Symbolics
from transform_util import transformBooleanToString, list2dict

W_1 = 0.2
W_2 = 0.8
epsilon = 0.1

import logging

def duplicate(s1,s2):
    compare = lambda a, b: len(a) == len(b) and len(a) == sum([1 for i, j in zip(a, b) if i == j])
    return compare(s1, s2)


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """
    Sparsemax building block: compute the threshold
    Parameters:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """
    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum
    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=-1):  # input will be modified
        """
        sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

sparsemax = SparsemaxFunction.apply

def init_logger(log_file=None):
    logger = logging.getLogger()
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def calc_True_Reward(action_sequence, qa_info, adaptive_flag = False, url=None):    
    try:
        symbolic_seq = list2dict(action_sequence)
        if len(symbolic_seq) == 0:
            return -1.0
        symbolic_exe = Symbolics(symbolic_seq, url=url)
        answer = symbolic_exe.executor(throw_exception=True)
    except: 
        return -1.0  # unvalid action sequence

    if adaptive_flag:
        return calc_adaptative_reward(answer, qa_info)
    else:
        return calc_01_reward(answer, qa_info)

def calc_01_reward(answer, qa_info):
    true_reward = 0.0
    response_entities = qa_info['response_entities'] if 'response_entities' in qa_info.keys() else []
    if type(response_entities) == type(''):
        response_entities = response_entities.strip().split('|')
    orig_response = qa_info['orig_response'].strip() if 'orig_response' in qa_info.keys() else ""
    qid = qa_info['state'].strip().replace(' ', '') if 'state' in qa_info.keys() else ""
    if qid.startswith("QuantitativeReasoning(Count)(All)") or qid.startswith("ComparativeReasoning(Count)(All)"):
        if not isinstance(answer, int):
            return -1.0
        if orig_response.isdigit():
            true_answer = int(orig_response)
        else:
            import re
            orig_response_temp = re.findall(r"\d+\.?\d*", orig_response)
            true_answer = sum([int(i) for i in orig_response_temp])
        if answer == true_answer:
            true_reward = 1.0
        return true_reward

    # For boolean, the returned answer is a list.
    elif qid.startswith("Verification(Boolean)(All)_") or qid.startswith("Verification(Boolean)(All)"):
        if answer == {}: true_reward = -1.0
        # To judge the returned answers are in dict format or boolean format.
        elif type(answer) == dict:
            temp = []
            if '|BOOL_RESULT|' in answer:
                temp.extend(answer['|BOOL_RESULT|'])
                predicted_answer_string = transformBooleanToString(temp)
                if predicted_answer_string != '' and predicted_answer_string == orig_response:
                    true_reward = 1.0
            else:
                true_reward = -1.0
        elif type(answer) == bool:
            predicted_answer = ""
            if answer:
                predicted_answer = "YES"
            elif not answer:
                predicted_answer = "NO"
            if predicted_answer == orig_response:
                true_reward = 1.0
        else:
            true_reward = -1.0
        return true_reward

    elif qid.startswith("SimpleQuestion(Direct)") or qid.startswith("LogicalReasoning(All)") or qid.startswith("QuantitativeReasoning(All)") or qid.startswith("ComparativeReasoning(All)"):
        # To judge the returned answers are in dict format or boolean format.
        if type(answer) == dict:
            if '|BOOL_RESULT|' in answer:
                return -1.0
            temp = []
            for key, value in answer.items():
                if key != '|BOOL_RESULT|' and value:
                    temp.extend(list(value))
            predicted_answer = temp

        elif type(answer) == type([]) or type(answer) == type(set([])):
            predicted_answer = sorted((list(answer)))
        elif type(answer) == int:
            return -1.0
            # predicted_answer = [answer]
        else:
            return -1.0
            # predicted_answer = [answer]
        # Solve the problem when response entities is [] and original response is 'None'.
        if orig_response == 'None':
            if len(predicted_answer) == 0:
                return 1.0
            else:
                return 0.0
        else:
            if len(response_entities) == 0:
                return 0.0
            else:
                if len(predicted_answer) == 0:
                    return 0.0
                else:
                    right_count = 0
                    for e in response_entities:
                        if (e in predicted_answer):
                            right_count += 1
                    p = float(right_count) / float(len(predicted_answer))
                    r = float(right_count) / float(len(response_entities))
                    if p == 0 and r == 0:
                        return 0.
                    f1 = 2 * p * r / (p + r)
                    return f1


# Adaptive reward: reward = R_type * (W_1 + W_2 * R_answer), W_1 + W_2 = 1;  W_1 = 0.2 W_2 = 0.8
def calc_adaptative_reward(answer, qa_info):
    response_entities = qa_info['response_entities'] if 'response_entities' in qa_info.keys() else []
    if type(response_entities) == type(''):
        response_entities = response_entities.strip().split('|')
    orig_response = qa_info['orig_response'].strip() if 'orig_response' in qa_info.keys() else ""
    qid = qa_info['state'].strip().replace(' ', '') if 'state' in qa_info.keys() else ""
    if qid.startswith("QuantitativeReasoning(Count)(All)") or qid.startswith("ComparativeReasoning(Count)(All)"):
        if not isinstance(answer, int):
            return 0.0
        R_type = 1.0
        if orig_response.isdigit():
            true_answer = int(orig_response)
        else:
            import re
            orig_response_temp = re.findall(r"\d+\.?\d*", orig_response)
            true_answer = sum([int(i) for i in orig_response_temp])
        # T: true_answer, P: predicted_answer, similarity s = 1-|T-P|/|T+P+ε|,ε is used to solve the problem when T or P is 0.
        R_answer = 1.0 - abs(float(true_answer - answer)) / abs(float(true_answer + answer + epsilon))
        return (R_type * (W_1 + W_2 * R_answer))

    # For boolean, the returned answer is a list.
    elif qid.startswith("Verification(Boolean)(All)_") or qid.startswith("Verification(Boolean)(All)"):
        if answer == {}: return -1.0
        # To judge the returned answers are in dict format or boolean format.
        elif type(answer) == dict:
            R_type = 1.0
            answer_list = []
            if '|BOOL_RESULT|' in answer:
                answer_list.extend(answer['|BOOL_RESULT|'])
                if len(answer_list) == 0:
                    return (R_type * W_1)
                else:
                    for i, item in enumerate(answer_list):
                        if item == True:
                            answer_list[i] = "YES"
                        elif item == False:
                            answer_list[i] = "NO"
                        else:
                            return 0.0
                    orig_response_list = orig_response.strip().split(' ')
                    true_answer_list = []
                    for token in orig_response_list:
                        if token == 'YES' or token == 'NO':
                            true_answer_list.append(token)
                    if len(true_answer_list) == 0:
                        return (R_type * W_1)
                    else:
                        if len(answer_list) <= len(true_answer_list):
                            correct_count=0.0
                            for i in range(len(answer_list)):
                                if answer_list[i] == true_answer_list[i]:
                                    correct_count+=1
                            R_answer = correct_count / float(len(true_answer_list))
                            return (R_type * (W_1 + W_2 * R_answer))
                        else:
                            # Expand the true_answer_list with duplicating the first element.
                            for i in range(len(answer_list)-len(true_answer_list)):
                                true_answer_list.append(true_answer_list[0])
                            correct_count = 0.0
                            for i in range(len(answer_list)):
                                if answer_list[i] == true_answer_list[i]:
                                    correct_count += 1
                            R_answer = correct_count / float(len(answer_list))
                            return (R_type * (W_1 + W_2 * R_answer))
        else:
            predicted_answer = ""
            if type(answer) == bool:
                if answer == True:
                    predicted_answer = "YES"
                elif answer == False:
                    predicted_answer = "NO"
                if predicted_answer == orig_response.strip():
                    return 1.0
                return (1.0 * W_1)
        return -1.0

    elif qid.startswith("SimpleQuestion(Direct)") or qid.startswith("LogicalReasoning(All)") or qid.startswith("QuantitativeReasoning(All)") or qid.startswith("ComparativeReasoning(All)"):
        # To judge the returned answers are in dict format or boolean format.
        R_type = 1.0
        if (type(answer) == dict):
            if '|BOOL_RESULT|' in answer:
                return -1.0
            temp = []
            for key, value in answer.items():
                if key != '|BOOL_RESULT|' and value:
                    temp.extend(list(value))
            predicted_answer = temp

        elif type(answer) == type([]) or type(answer) == type(set([])):
            predicted_answer = sorted((list(answer)))
        elif type(answer) == int:
            return -1.0
            # predicted_answer = [answer]
        else:
            return -1.0
            # predicted_answer = [answer]
        # Solve the problem when response entities is [] and original response is 'None'.
        if orig_response == 'None' and len(response_entities) == 0:
            if len(predicted_answer) == 0:
                return 1.0
            else:
                return R_type * W_1
        else:
            if len(response_entities) == 0:
                return R_type * W_1
            else:
                if len(predicted_answer) == 0:
                    return R_type * W_1
                else:
                    right_count = 0
                    for e in response_entities:
                        if (e in predicted_answer):
                            right_count += 1
                    # Compute F1 value as reward.
                    precision = float(right_count)/float(len(predicted_answer)) if len(predicted_answer) != 0 else 0
                    recall = float(right_count)/float(len(response_entities)) if len(response_entities) != 0 else 0
                    F1 = 2 * precision * recall / (recall + precision) if (precision!=0 and recall!=0) else 0
                    return (R_type * (W_1 + W_2 * F1))

# Compute proximity for Curriculum-guided Hindsight Experience Replay.
def calculate_proximity(action_tokens, action_buffer):
    max_proximity = 0.0
    if action_buffer is None:
        return max_proximity
    else:
        for action_in_buffer in action_buffer:
            proximity = levenshtein_similarity(action_tokens, action_in_buffer)
            if proximity > max_proximity:
                max_proximity = proximity
        return max_proximity

# Compute diversity for Curriculum-guided Hindsight Experience Replay.
def calculate_diversity(action_tokens, action_buffer):
    beta = 1.0
    similarity_sum = 0.0
    if action_buffer is None:
        return beta - similarity_sum
    else:
        for action_in_buffer in action_buffer:
            similarity_sum += levenshtein_similarity(action_tokens, action_in_buffer)
        diversity = max(beta - similarity_sum / float(len(action_buffer)), 0.0)
        return diversity

def levenshtein_similarity(source, target):
    """
    To compute the edit-distance between source and target.
    If source is list, regard each element in the list as a character.
    :param list1
    :param list2
    :return:
    """
    if source is None or len(source) == 0:
        return 0.0
    elif target is None or len(target) == 0:
        return 0.0
    elif type(source) != type(target):
        return 0.0
    matrix = [[i + j for j in range(len(target) + 1)] for i in range(len(source) + 1)]
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if source[i - 1] == target[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    distance = float(matrix[len(source)][len(target)])
    length = float(len(source) if len(source) >= len(target) else len(target))
    return 1.0 - distance / length
