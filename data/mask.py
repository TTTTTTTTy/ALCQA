import os
import json 
import random
import Levenshtein
from tqdm import tqdm
import unicodedata
from pattern.text.en import singularize

def strip_accents(text):
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')


def find_most_similar_segment(lst, s):
    max_similarity = 0.
    res = (-1, -1)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            s1 = ' '.join(lst[i:j+1])
            if 'how ' in s1  or 'what ' in s1 or 'which ' in s1 or  ' many ' in s1 or s1.startswith('many ') or 'ENTITY' in s1 or 'TYPE' in s1 or 'INT' in s1:
                continue
            if s1.endswith('s'):
                s1 = singularize(s1)
            sim = Levenshtein.ratio( s1, s)
            if sim > max_similarity:
                max_similarity = sim
                res = (i, j+1)
    return res, max_similarity

def sort(lst, prefix):
    d = {}
    for idx, act in enumerate(lst):
        if act.startswith(prefix):
            if act not in d:
                d[act] = '%s%d' % (prefix, len(d)+1)
            lst[idx] = d[act]

def shuffle_vocab(d):
    '''
    shuffle d['mask'] and update action
    '''
    vocab = d['mask']
    entitys = [(k, v)  for k, v in vocab.items() if k.startswith('E')]
    types = [ (k, v)  for k, v in vocab.items() if k.startswith('T')]
    mask = {}
    if len(entitys) > 1:
        idx_lst = [i for i in range(len(entitys))]
        random.shuffle(idx_lst)
        for i in range(len(entitys)):
            vocab['ENTITY%d' % (i+1)] = entitys[idx_lst[i]][1]
            mask[entitys[idx_lst[i]][0]] = 'ENTITY%d' % (i+1)
    if len(types) > 1:
        idx_lst = [i for i in range(len(types))]
        random.shuffle(idx_lst)
        for i in range(len(types)):
            vocab['TYPE%d' % (i+1)] = types[idx_lst[i]][1]
            mask[types[idx_lst[i]][0]] = 'TYPE%d' % (i+1)
    origin_mask_name = d['mask_name'].copy()
    for k in d['mask_name'].keys():
        if k in mask:
            d['mask_name'][k] = origin_mask_name[mask[k]]
    action = d['action'].split()
    for idx, act in enumerate(action):
        if act in mask:
            action[idx] = mask[act]
    d['action'] =  ' '.join(action)


def mask_sent(question, mask):
    question_lst = question.lower().split(' ')
    lst = [(k, strip_accents(v)) for k, v in mask.items()]
    lst.sort(key=lambda x: len(x[1]), reverse=True)
    is_first = True
    while True:
        find_lst = []
        for m, vocab in lst:
            if m.startswith('RELATION'):
                continue
            (start, end), sim = find_most_similar_segment(question_lst, vocab.lower())
            if  sim > 0.9 - 0.05 * len(vocab.split(' ')) and sim > 0.3:
                find_lst.append(((start,end), sim, m))
                # question_lst[start: end] = [m]
            # elif is_first:
            #     print(m, vocab)
            #     print(question)
        if len(find_lst) == 0:
            break
        find_lst.sort(key=lambda x: x[1], reverse=True)
        overlap = [False for _ in range(len(question_lst))]
        for (start, end), sim, m in find_lst:
            if True in overlap[start:end]:
                continue
            question_lst[start: end] = [m] + ['[PAD]' for _ in range(end-start-1)]
            for idx in range(start,end):
                overlap[idx] = True
        question_lst = [ token for token in question_lst if token != '[PAD]' ]
        is_first = False
    return ' '.join(question_lst)



if __name__=='__main__':
    for data_dir in ['pretrain', 'rl', 'test/test_sample']:    
        print(data_dir)
        data_path = os.path.join(data_dir, 'data.json') 
        out = open(os.path.join(data_dir, 'data_masked.json'), 'w')

        for line in tqdm(open(data_path).readlines()):
            d = json.loads(line)
            masked = mask_sent(d['question'], d['mask_name'])
            d['masked_question'] = masked

            out.write(json.dumps(d) + '\n')