import warnings
warnings.filterwarnings('ignore')
import sys
import os
import torch
import re
import json
import numpy as np
import argparse
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from nltk.stem import WordNetLemmatizer
import unicodedata
from tqdm import tqdm
from pattern.en import lemma

def strip_accents(text):
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')

# wnl = WordNetLemmatizer()
def diff(text1, text2, action, mask_name, cur_text=None):
    mask_name = {k:strip_accents(v).lower() for k, v in mask_name.items()}
    mask_vocabs = [act for act in action.split(' ') if act in mask_name]
    uncovered = [ act for act in mask_name.keys() if act not in mask_vocabs]
    tokens1 = text1.split(' ')
    tokens2 = text2.split(' ')
    # lemmaed_tokens1 = [wnl.lemmatize(token) for token in tokens1]
    # lemmaed_tokens2 = [wnl.lemmatize(token) for token in tokens2]
    lemmaed_tokens1 = [lemma(token) for token in tokens1]
    lemmaed_tokens2 = [lemma(token) for token in tokens2]
    uncovered_mask_tokens = set()
    for mask in uncovered:
        for token in mask_name[mask].split():
            uncovered_mask_tokens.add(token)
    covered_mask_tokens = set()
    for mask in mask_vocabs:
        for token in mask_name[mask].split():
            covered_mask_tokens.add(token)
    diff_res = []
    for idx, token in enumerate(tokens2):
        # if token in tokens1 and len(diff_res) > 0:
        #     break
        if lemmaed_tokens2[idx] not in lemmaed_tokens1:
            if token not in uncovered_mask_tokens:
                diff_res.append(token)
        else:
            lemmaed_tokens1.remove(lemmaed_tokens2[idx])
    if cur_text is not None:
        diff_res_limited = []
        for token in diff_res:
            if token not in covered_mask_tokens:
                diff_res_limited.append(token)
        return ' '.join(diff_res_limited) + ' ' + cur_text
    return ' '.join(diff_res)

mask_dict = {1: 'x', 2: 'y', 3:'z'}
'''
format like
[
    ["A1 ( ENTITY1 RELATION1 TYPE1 )"], 
    ["A1 ( ENTITY1 RELATION1 TYPE1 )", "A9 ( ENTITY2 RELATION1 TYPE1 )"],
    ["A1 ( ENTITY1 RELATION1 TYPE1 )", "A9 ( ENTITY2 RELATION1 TYPE1 )", "A11 ( )"]
]
'''
def get_action_texts(actions, d):
    action_texts = []
    cur_action_texts = []
    for action_list in actions:
        action_list_list = [lst.split(' ')  for lst in action_list]
        mask = d['mask_name']
        action_text = []
        a2_count = 0
        for idx, action_list in enumerate(action_list_list):
            assert action_list[0].startswith('A')
            is_a2 = not np.sum([l[0] != 'A2' for l in action_list_list[idx:]]) 
            temp_action_text = []
            if is_a2:
                temp_action_text.append('<a1>')
                a2_count += 1
            else:
                temp_action_text.append('<%s>' %  action_list[0].lower())
            for act in action_list[1:]:
                if act in mask:
                    if is_a2 and temp_action_text[-1] != '<a1>' and act.startswith('TYPE'):  # the second type of a2 action
                        temp_action_text.append('<entity>')
                        temp_action_text.append(mask_dict[a2_count])
                        temp_action_text.append('</entity>')
                    else:
                        temp_action_text.append('<%s>' %  re.sub(r'[0-9]', '', act).lower())
                        temp_action_text.append(mask[act].lower())
                        temp_action_text.append('</%s>' %  re.sub(r'[0-9]', '', act).lower())
                else:
                    if act == '(':
                        continue
                    elif act == ')':
                        if is_a2:
                            temp_action_text.append('</a1>')
                        else:
                            temp_action_text.append('</%s>' %  action_list[0].lower())
                    else:
                        temp_action_text.append('<%s>' %  act.lower())
            action_text.extend(temp_action_text)
        if len(action_list_list) > 1 and action_list_list[-1][0] in ['A1', 'A2', 'A8', 'A9', 'A10']:
            cur_action_texts.append(' '.join(temp_action_text).replace(action_list_list[-1][0].lower(), 'a1'))
        else:
            cur_action_texts.append(None)
        action_text = ' '.join(action_text)
        action_texts.append(action_text)
    return action_texts, cur_action_texts


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Name of trained action2text model")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base/')
    added_tokens = ['<entity>', '</entity>', '<relation>', '</relation>', '<type>', '</type>', '<int>', '</int>', '<->', '<&>', '<|>']
    for i in range(16):
        added_tokens.append('<a%d>' % (i+1))
        added_tokens.append('</a%d>' % (i+1))
    tokenizer.add_tokens(added_tokens)
    config = BartConfig.from_pretrained('facebook/bart-base/')
    config.forced_bos_token_id = None
    model = BartForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model_dict = torch.load(os.path.join('saved_models/action2text', args.model_name))
    model.load_state_dict(model_dict)
    model.to(device)

    out = open('data/pretrain/decomposed.txt', 'w')
    lines = open('data/pretrain/data.json').readlines()
    for ex_idx, line in tqdm(enumerate(lines), total=len(lines)):
        d = json.loads(line)
        sub_questions = []
        action_lst = d['action'].split(') ')
        action_lst = [act + ')' if not act.endswith(')') else act  for act in action_lst]
        action_lst_refined = [action_lst[0]]
        for i in range(1, len(action_lst)):
            lst = [act for act in action_lst[i].split(' ') if act.startswith('A') or act.startswith('E') or act.startswith('R') or act.startswith('T')]
            last_lst = action_lst_refined[-1].split(' ')
            if len(lst) != len(last_lst):
                action_lst_refined.append(action_lst[i])
            else:
                for a1, a2 in zip(lst, last_lst):
                    if a1 != a2:
                        action_lst_refined.append(action_lst[i])
                        break
        actions = [ action_lst_refined[:i+1] for i in range(len(action_lst_refined))]
            
        action_texts, cur_action_texts = get_action_texts(actions, d)
        inputs = tokenizer(action_texts, return_tensors='pt', padding=True)
        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'].to(device), num_beams=3, min_length=5, max_length=100)
        texts = []
        for idx, (action_list, action_text, g) in enumerate(zip(actions, action_texts, summary_ids)):
            text = tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            action_list_list = [lst.split(' ')  for lst in action_list]
            A2_num = 0
            for i in range(len(action_list_list)-1, -1, -1):
                if action_list_list[i][0] != 'A2':
                    break
                A2_num += 1
            for i in range(len(action_list_list)-1, -1, -1):
                if A2_num == 0:
                    break
                text = text.replace(' %s ' % (mask_dict[A2_num]),' which %s ' % d['mask_name'][action_list_list[i][-2]])
                A2_num -= 1
                
            texts.append(text)
            if idx > 0:
                sub_question = diff(texts[-2], texts[-1], action_list[-1], d['mask_name'],)
            else:
                sub_question = texts[-1]
            sub_questions.append(sub_question)
        out.write(' # '.join(sub_questions) + '\n')
        out.flush()