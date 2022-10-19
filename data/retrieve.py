import Levenshtein
import json
from tqdm import tqdm
import re

def sort(masked_question, prefix):
    d = {}
    for token in masked_question.split(' '):
        if token.startswith(prefix) and token not in d:
            d[token] = '%s%d' % (prefix, len(d)+1)
    return d

def update_d(d, sorted_d):
    new_question = []
    for token in d['masked_question'].split(" "):
        if token not in sorted_d:
            new_question.append(token)
        else:
            new_question.append(sorted_d[token])
    new_question = ' '.join(new_question)
    new_vocab = {}
    for k, v in d['mask'].items():
        if k not in sorted_d:
            new_vocab[k] = v
        else:
            new_vocab[sorted_d[k]] = v
    return new_question, new_vocab

def detect(tokens, triggers):
    for t in triggers:
        if t in tokens:
            return t
    return None

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
    triggers = [('min','max'), ('less', 'more', 'same'), ('lesser', 'more', 'same'),  ('atmost', 'atleast', 'exactly', 'around'), \
         ('atmost', 'atleast', 'exactly', 'approximately'), ('less', 'greater', 'same'), ('lesser', 'greater', 'same'), ('or', 'but not', 'and')]
    for trigger in triggers:
        t1 = detect(source, trigger)
        t2 = detect(target, trigger)
        if t1 != None and t2 != None and t1 != t2:
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

def unmask_question(question):
    token_lst = []
    for token in question.split(' '):
        if token.startswith('ENTITY'):
            token_lst.append('ENTITY')
        elif token.startswith('TYPE'):
            token_lst.append('TYPE')
        else:
            token_lst.append(token)
    return ' '.join(token_lst)

def get_feat(mask_d):
    entity_count, relation_count, type_count, int_count = 0, 0, 0, 0
    for mask in mask_d.keys():
        if mask.startswith('ENTITY'):
            entity_count += 1
        elif mask.startswith('TYPE'):
            type_count += 1
        elif mask.startswith('REL'):
            relation_count += 1
        else:
            int_count += 1
    feat = '%d_%d_%d_%d' % (entity_count, relation_count, type_count, int_count)
    return feat

def load_memory(path, memory=None):
    if memory == None:
        memory = {}
    for line in open(path).readlines():
        d = json.loads(line)
        entities = [x for x in d['mask'].keys() if x.startswith('ENTITY')]
        entity_num = len(entities)
        types = [x for x in d['mask'].keys() if x.startswith('TYPE')]
        type_num = len(types)
        sorted_d = sort(d['masked_question'], 'ENTITY')
        if len(sorted_d) < entity_num - 1:  # we canot map two entities, skip
            continue
        for k in entities:
            if k not in sorted_d:
                sorted_d[k] = 'ENTITY%d' % (len(sorted_d) + 1)
        temp = sort(d['masked_question'], 'TYPE')
        if len(temp) < type_num - 1:
            continue
        for k in types:
            if k not in temp:
                temp[k] = 'TYPE%d' % (len(temp) + 1)
        sorted_d.update(temp)
        new_question, new_vocab = update_d(d, sorted_d)
        feat = get_feat(d['mask'])
        if feat not in memory:
            memory[feat] = []
        # d['state'] = re.sub('[0-9]+|_| ', '', d['state'])
        d['mask'] = new_vocab
        del d['mask_name']
        d['masked_question'] = new_question
        memory[feat].append((d))
    return memory

def get_most_sim_n(memory, feat, tokens, state, n):
    temp_memory = memory[feat] if feat in memory else None
    if temp_memory == None:
        return []
    score = []
    for d in temp_memory:
        # if d['masked_question'] == ' '.join(tokens) and set(vocab.values()) == set(d['mask'].values()): # same question
        #     continue
        if d['state'] == state:  # same question
            continue
        if re.sub('[0-9]+|_| ', '', d['state']).replace('pretrain', '') != re.sub('[0-9]+|_| ', '', state): # not same question type
            continue
        sim = levenshtein_similarity(d['masked_question'].split(' '), tokens)
        if sim > 0.6:
            d['weight'] = sim
            score.append(d)
    sorted_score = sorted(score, key=lambda x : x['weight'], reverse=True)
    return sorted_score[:n]

if __name__=='__main__':
    memory = load_memory('pretrain/data_masked.json')
    memory = load_memory('rl/data_masked.json', memory)
    print( 'load %d questions' % (sum([len(memory[k]) for k in memory.keys() ])))
    result = []
    vocab_dict = []
    for line in tqdm(open('test/test_sample/data_masked.json').readlines()):
        d = json.loads(line)
        lst = []
        entities = [x for x in d['mask'].keys() if x.startswith('ENTITY')]
        entity_num = len(entities)
        types = [x for x in d['mask'].keys() if x.startswith('TYPE')]
        type_num = len(types)
        sorted_d = sort(d['masked_question'], 'ENTITY')
        if len(sorted_d) < entity_num - 1:  # we canot map two entities, skip
            result.append(lst)
            vocab_dict.append({})
            continue
        for k in entities:
            if k not in sorted_d:
                sorted_d[k] = 'ENTITY%d' % (len(sorted_d) + 1)
        temp = sort(d['masked_question'], 'TYPE')
        if len(temp) < type_num - 1:
            result.append(lst)
            vocab_dict.append({})
            continue
        for k in types:
            if k not in temp:
                temp[k] = 'TYPE%d' % (len(temp) + 1)
        sorted_d.update(temp)
        vocab_dict.append(sorted_d)
        new_question, new_vocab = update_d(d, sorted_d)
        lst = get_most_sim_n(memory, get_feat(d['mask']), new_question.split(), d['state'], 10)
        result.append(lst)
    assert len(result) == len(vocab_dict)
    json.dump(result, open('test/test_sample/most_sim_10.json', 'w'), ensure_ascii=False, indent=4)
    json.dump(vocab_dict, open('test/test_sample/vocab_dict.json', 'w'), ensure_ascii=False, indent=4)
