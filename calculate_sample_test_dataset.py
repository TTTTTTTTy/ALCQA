import sys
from symbolics import Symbolics
from transform_util import transformBooleanToString, list2dict

import os
import re
import json
import logging
import sys
import copy
from utils import init_logger

def transMask2Action(json_path, action_path, state, logger, output_path, global_states, answer_memory):
    dataset = []
    with open(json_path) as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
    predict_actions = []
    with open(action_path) as f:
        for line in f.readlines():
            # valid, action = line.strip().split('\t')
            # if valid:
            #     predict_actions.append(action)
            # else:
            #     predict_actions.append('')
            predict_actions.append(line.strip())
  
    assert len(dataset) == len(predict_actions)
    num = 0
    # return entity
    total_precision = 0
    total_recall = 0
    total_right_count = 0
    total_answer_count = 0
    total_response_count = 0
    # return bool
    bool_right_count = 0
    # return num
    count_right_count = 0
    tmp_count = 0
    for x, action in zip(dataset, predict_actions):
        qid = x['state']
        response_entities = x["response_entities"].strip() if x["response_entities"] is not None else ""
        response_entities = response_entities.strip().split("|")
        orig_response = x["orig_response"].strip() if x["orig_response"] is not None else ""
        if qid.startswith(state):
            # tmp_count += 1
            # if tmp_count == 100:
            #     break
            mask = x['mask']
            new_action = [ mask[act] if act in mask else act for act in action.split()]
            num += 1
            # if num > 10:
            #     break
            logger.info("%d: %s", num, qid)
            logger.info('question: %s', x['question'])
            logger.info('action_mask: %s', action)
            logger.info('action: %s', ' '.join(new_action))
            if x['state'] not in answer_memory:
                answer_memory[x['state']] = {}
            if ' '.join(new_action) not in answer_memory[x['state']]:
                symbolic_seq = list2dict(new_action)
                logging.info(symbolic_seq)
                symbolic_exe = Symbolics(symbolic_seq)
                answer = symbolic_exe.executor()
                tmp_answer = copy.deepcopy(answer)
                if type(tmp_answer) == dict:
                    for k in tmp_answer.keys():
                        tmp_answer[k] = list(tmp_answer[k])
                answer_memory[x['state']][' '.join(new_action)] = tmp_answer
            else:
                tmp_answer = answer_memory[x['state']][' '.join(new_action)]
                answer = copy.deepcopy(tmp_answer)
                if type(answer) == dict:
                    for k in answer.keys():
                        if k.startswith('Q'):
                            answer[k] = set(answer[k])

            if state.startswith("QuantitativeReasoning(Count)(All)") or state.startswith("ComparativeReasoning(Count)(All)"):
                logger.info("answer:%s, orig_response:%s", answer, orig_response)
                if orig_response.isdigit() and answer == int(orig_response):
                    count_right_count += 1
                    logger.info("count_right_count+1")
                else:
                    orig_response = re.findall(r"\d+\.?\d*", orig_response)
                    orig_response = sum([int(i) for i in orig_response])
                    if answer == orig_response:
                        count_right_count += 1
                        logger.info("count_right_count+1")
            # For boolean, the returned answer is a list.
            elif state.startswith("Verification(Boolean)(All)"):
                # To judge the returned answers are in dict format or boolean format.
                if type(answer) == dict:
                    temp = []
                    if '|BOOL_RESULT|' in answer:
                        temp.extend(answer['|BOOL_RESULT|'])
                        answer = temp
                        answer_string = transformBooleanToString(answer)
                        logger.info("answer:%s, orig_response:%s", answer_string, orig_response)
                        if answer_string!='' and answer_string == orig_response:
                            bool_right_count += 1
                            logger.info("bool_right_count+1")
                else:
                    if answer:
                        answer = "YES"
                    if not answer:
                        answer = "NO"
                    logger.info("answer:%s, orig_response:%s", answer, orig_response)
                    if answer == orig_response:
                        bool_right_count += 1
                        logger.info("bool_right_count+1")

            else:
                # To judge the returned answers are in dict format or boolean format.
                if type(answer) == dict:
                    temp = []
                    if '|BOOL_RESULT|' in answer:
                        temp.extend(answer['|BOOL_RESULT|'])
                    else:
                        for key, value in answer.items():
                            if (value):
                                temp.extend(list(value))
                    answer = temp

                elif type(answer) == type([]) or type(answer) == type(set([])):
                    answer = sorted((list(answer)))
                elif type(answer) == int:
                    answer = [answer]
                else:
                    answer = [answer]

                right_count = 0
                for e in response_entities:
                    if e in answer:
                        right_count += 1
                total_right_count += right_count
                total_answer_count += len(answer)
                total_response_count += len(response_entities)
                precision = right_count / float(len(answer)) if len(answer) != 0 else 0
                total_precision += precision
                recall = (right_count / float(len(response_entities))) if len(response_entities) != 0 else 0
                total_recall += recall
                logger.info("orig:%d, answer:%d, right:%d", len(response_entities), len(answer), right_count)
                logger.info("Precision:%f", precision)
                logger.info("Recall:%f", recall)
            logger.info("============================")

    output_path.write(state + '\n')
    if state.startswith("QuantitativeReasoning(Count)(All)") or state.startswith("ComparativeReasoning(Count)(All)"):
        output_path.write("count_right_count: %d\n" %count_right_count)
        output_path.write("total_num: %d\n" %num)
        if num > 0:
            output_path.write("precision: %.4f\n" % (float(count_right_count) / num))
        global_states['total_precision'] += float(count_right_count)
        global_states['total_recall'] += float(count_right_count)
        global_states['question_count'] += num
        global_states['total_f1'] += float(count_right_count) / num

    elif state.startswith("Verification(Boolean)(All)"):
        output_path.write("bool_right_count: %d\n" %bool_right_count)
        output_path.write("total_num: %d\n" %num)
        if num > 0:
            output_path.write("precision: %.4f\n" % (float(bool_right_count) / num))
        global_states['total_precision'] += float(bool_right_count)
        global_states['total_recall'] += float(bool_right_count)
        global_states['question_count'] += num
        global_states['total_f1'] += float(bool_right_count) / num

    else:
        mean_pre = total_precision / num if num != 0 else 0.0
        mean_recall = total_recall / num if num != 0 else 0.0
        mean_f1 = 2 * (mean_pre * mean_recall) / (mean_pre + mean_recall) if mean_pre > 0 or mean_recall > 0 else 0.
        mean_pre2 = float(total_right_count) / total_answer_count if total_answer_count!=0 else 0.0
        mean_recall2 = float(total_right_count) / total_response_count if total_response_count!=0 else 0.0
        mean_f12 = 2 * (mean_pre2 * mean_recall2) / (mean_pre2 + mean_recall2) if mean_pre2 > 0 or mean_recall2 > 0 else 0.
        output_path.write("total_num::total_right::total_answer::total_response -> %d::%d::%d::%d\n" \
                        % (num, total_right_count, total_answer_count, total_response_count))
        output_path.write("mean_pre::mean_recall::mean_f1 -> %.4f::%.4f::%.4f\n"  % (mean_pre, mean_recall, mean_f1))
        output_path.write("mean_pre2::mean_recall2::mean_f12 -> %.4f::%.4f::%.4f\n" % (mean_pre2, mean_recall2, mean_f12))
        global_states['total_precision'] += total_precision
        global_states['total_recall'] += total_recall
        global_states['question_count'] += num
        global_states['total_f1'] += mean_f1
        
    output_path.write('++++++++++++++\n\n')
    output_path.flush()

def calculate_RL_or_DL_result(json_path, action_path, logger, output_path, answer_memory):
    global_states = {'total_precision':0., 'total_recall':0., 'question_count':0, 'total_f1': 0.}
    fw = open(output_path, 'w', encoding="UTF-8")
    # state_list = ["Verification(Boolean)(All)"]
    state_list = ["Verification(Boolean)(All)", "QuantitativeReasoning(Count)(All)",
                  "QuantitativeReasoning(All)", "ComparativeReasoning(Count)(All)", "ComparativeReasoning(All)",
                  "LogicalReasoning(All)", "SimpleQuestion(Direct)"]
    # state_list = ["Verification(Boolean)(All)"]
    for state in state_list:
        transMask2Action(json_path, action_path, state, logger, fw, global_states, answer_memory)
    macro_f1 = global_states['total_f1'] / len(state_list)
    mean_pre = global_states['total_precision'] / global_states['question_count']
    mean_recall = global_states['total_recall'] / global_states['question_count']
    micro_f1 = 2 * (mean_pre * mean_recall) / (mean_pre + mean_recall) 
    fw.write('macro_f1: %.4f\n' % macro_f1)
    fw.write('micro_f1: %.4f\n' % micro_f1)
    fw.flush()

    fw.close()


if __name__ == "__main__":
    test_dir = 'test_sample'
    name = sys.argv[1]
    log_file = 'data/test/%s/results/logs/%s.log' % (test_dir, name)
    if os.path.exists(log_file):
        os.remove(log_file)
    logger = init_logger(log_file)
    logger.info('use dataset: [%s], test file: [%s]' % (test_dir, name))
    answer_memory = {}
    if os.path.exists('data/test/%s/answer_memory.json' % test_dir):
        answer_memory = json.load(open('data/test/%s/answer_memory.json'% test_dir))
        logger.info('loaded answer memory')
    calculate_RL_or_DL_result(json_path='data/test/%s/data.json' % test_dir, 
                            action_path='data/test/%s/results/%s.txt' %  (test_dir, name), 
                            logger=logger, 
                            output_path='data/test/%s/results/%s_p.txt' %  (test_dir, name),
                            answer_memory=answer_memory)
    json.dump(answer_memory, open('data/test/%s/answer_memory.json' % test_dir, 'w'))
