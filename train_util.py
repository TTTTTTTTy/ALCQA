import os
import re
import json
import math
import random
from random import randrange
from torch.serialization import save
from tqdm import tqdm
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.stem import WordNetLemmatizer

from utils import calc_True_Reward, duplicate, calculate_proximity, calculate_diversity

ALPHA = 0.1
ETA = 0.08
LAMBDA_0 = 0.1
MAX_MEMORY_BUFFER_SIZE = 10

MEMORY_USE_COUNT = 0
MEMORY_NOT_USE_COUNT = 0

FINETUNE_DATASET = {}
UPDATE_FLAG = set()

wnl = WordNetLemmatizer()
def diff(text1, text2):
    tokens1 = text1.lower().split(' ')
    tokens2 = text2.lower().split(' ')
    lemmaed_tokens1 = [wnl.lemmatize(token) for token in tokens1]
    lemmaed_tokens2 = [wnl.lemmatize(token) for token in tokens2]
    diff_res = []
    for idx, token in enumerate(tokens2):
        # if token in tokens1 and len(diff_res) > 0:
        #     break
        if lemmaed_tokens2[idx] not in lemmaed_tokens1:
            diff_res.append(token)
        else:
            lemmaed_tokens1.remove(lemmaed_tokens2[idx])
    return ' '.join(diff_res)

def compute_loss(scores, target, ignore_index):
    """
    scores (FloatTensor): (tgt_s_num*tgt_s_len*batch, n_vocab)
    align (LongTensor): (tgt_s_num*tgt_s_len*batch, seq_len]
    target (LongTensor): (batch_size*tgt_len)
    """
    # probabilities assigned by the model to the gold targets
    vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)
    vocab_probs = vocab_probs.clamp(min=1e-4)  # prevent nan error
    loss = -vocab_probs.log() 
    # Drop padding.
    loss[target == ignore_index] = 0
    return loss


def load_data(data_path, args, load_emb=True):
    dataset = []
    data = open(os.path.join(data_path, 'data.json')).readlines()
    ques_emb, vocab_emb = None, None
    if load_emb:
        ques_emb, vocab_emb = [], []
        vocab = open(os.path.join(data_path, 'embedding', 'vocab.txt')).readlines()
        ques_emb_file = open( os.path.join(data_path, 'embedding', 'decomposed_predict.bin'), 'rb')
        # ques_emb_file = open( os.path.join(data_path, 'embedding', 'question.bin'), 'rb')
        vocab_emb_file = open( os.path.join(data_path, 'embedding', 'vocab.bin'), 'rb')
        # for qa, v, sim_d, vocab_d in zip(data, vocab, most_sim_instances, vocab_dicts):
        for qa, v in zip(data, vocab):
            ques_emb.append( torch.from_numpy(np.load(ques_emb_file))) 
            vocab_emb.append( torch.from_numpy(np.load(vocab_emb_file)))
            dataset.append(json.loads(qa))
            dataset[-1]['vocab'] = v.strip().split('|')
            dataset[-1]['length'] = ques_emb[-1].size(0)
            # dataset[-1]['sim_d'] = sim_d
            # dataset[-1]['vocab_dict'] = vocab_d
        if args.assistant_file is not None:
            most_sim_instances = json.load(open(os.path.join(data_path, args.assistant_file)))
            vocab_dicts = json.load(open(os.path.join(data_path, 'vocab_dict.json')))
            assert len(data) == len(vocab) == len(most_sim_instances) == len(vocab_dicts)
            for idx, (sim_d, vocab_d) in enumerate(zip(most_sim_instances, vocab_dicts)):
                dataset[idx]['sim_d'] = sim_d[:args.sim_num]
                dataset[idx]['vocab_dict'] = vocab_d
    else:
        for qa in data:
            dataset.append(json.loads(qa))
    return dataset, ques_emb, vocab_emb


def batch_data(device, dataset, ques_emb, vocab_emb, symbol_list, symbol_dict, start_idx, end_idx, pad_idx, embedding=None, type_embedding=None, word2type=None):
    assert len(dataset) == len(ques_emb) == len(vocab_emb)
    # sort in descending order with question lengths
    idx_dict = { idx:dataset[idx]['length'] for idx in range(len(dataset))}
    sorted_dict = sorted(idx_dict.items(), key=lambda x: x[1], reverse=True)
    reverse_dict = {idx:i   for i, (idx, _) in enumerate(sorted_dict)}
    dataset = [dataset[idx] for idx, _ in sorted_dict]
    ques_emb = [ques_emb[idx] for idx, _ in sorted_dict]
    vocab_emb = [vocab_emb[idx] for idx, _ in sorted_dict]

    # pad question embedding to same length
    lengths = [ d['length'] for d in dataset]
    max_length = lengths[0]
    src = []
    for emb in ques_emb:
        src.append(F.pad(input=emb, pad=(0, 0, 0, max_length-emb.size(0)), value=0))
    src = torch.stack(src, dim=0).to(device)
    lengths = torch.tensor(lengths).to(device)
    # get target embedding
    padded_target = None
    padded_target_emb = None
    if 'action' in dataset[0]:
        type_lst, target, target_emb = [], [], []
        max_total_action_len = 0
        for example, vocab_e in zip(dataset, vocab_emb):
            vocab_e = vocab_e.to(device)
            temp_type, temp_target, temp_emb = [0], [start_idx], [embedding(torch.tensor([start_idx]).to(device))]
            vocab_dict = {}
            for v_id, vocab in enumerate(example['vocab']):
                vocab_dict[vocab] = v_id
            for act in example['action'].split():
                if act in symbol_dict:
                    s_id = symbol_dict[act]
                    emb = embedding(torch.tensor([s_id]).to(device))
                    temp_emb.append(emb)
                    temp_target.append(s_id)
                    temp_type.append(word2type[s_id])
                else: # ent, rel, type, etc
                    temp_emb.append(vocab_e[vocab_dict[act]].unsqueeze(0))
                    temp_target.append(vocab_dict[act] + len(symbol_dict))
                    if act.startswith('E'):
                        temp_type.append(word2type[len(symbol_dict)])
                    elif act.startswith('R'):
                        temp_type.append(word2type[len(symbol_dict)+1])
                    elif act.startswith('T'):
                        temp_type.append(word2type[len(symbol_dict)+2])
                    else:
                        temp_type.append(word2type[len(symbol_dict)+3])
            # add eos symbol
            temp_target.append(end_idx)
            temp_emb.append(embedding(torch.tensor([end_idx]).to(device)))
            temp_type.append(word2type[end_idx])
            target.append(torch.tensor(temp_target))
            target_emb.append(torch.cat(temp_emb, dim=0))
            type_lst.append(torch.tensor(temp_type))
            if len(example['action'].split()) + 2 > max_total_action_len:  #  <start>,<end>
                max_total_action_len = len(example['action'].split()) + 2

        # pad examples to same length
        padded_target = torch.full((len(dataset), max_total_action_len), pad_idx)
        for idx, c_target in enumerate(target):  # [tensor([]), tensor([]), ...]
            padded_target[idx, :c_target.size(0)] = c_target
        padded_target = padded_target.to(device)

        padded_target_emb = torch.zeros((len(dataset), max_total_action_len, vocab_emb[0].size(-1) + type_embedding.embedding_dim)).to(device)
        for idx, c_target_emb in enumerate(target_emb):  # [tensor([]), tensor([]), ...] [seq_len x hidden]
            type_emb = type_embedding(type_lst[idx].to(device))
            assert type_emb.size(0) == c_target_emb.size(0)
            c_target_emb = torch.cat([c_target_emb, type_emb], dim=-1)
            padded_target_emb[idx, :c_target_emb.size(0)] = c_target_emb 

    max_vocab_num = max([emb.size(0) for emb in vocab_emb])
    cand_emb = torch.zeros((len(vocab_emb), max_vocab_num, vocab_emb[0].size(-1)))
    for idx, emb in enumerate(vocab_emb):
        cand_emb[idx, :emb.size(0)] = emb
    cand_emb = cand_emb.to(device)

    vocab_size = []
    words = [] #idx2word, for predict
    type_list = []
    
    for idx, example in enumerate(dataset):
        words.append([])
        type_list.append([])
        words[-1].extend(symbol_list)
        words[-1].extend(example['vocab'])
        vocab_size.append(len(example['vocab']))
        for v in example['vocab']:
            if v.startswith('E'):
                type_list[-1].append(word2type[len(symbol_dict)])
            elif v.startswith('R'):
                type_list[-1].append(word2type[len(symbol_dict)+1])
            elif v.startswith('T'):
                type_list[-1].append(word2type[len(symbol_dict)+2])
            else:  # INT
                type_list[-1].append(word2type[len(symbol_dict)+3])
    vocab_size = torch.tensor(vocab_size).to(device)
    return src, lengths, padded_target, padded_target_emb, cand_emb, vocab_size, words, type_list, reverse_dict


def unpack_ids(ids, start_idx, end_idx):
    unpack = []
    eos = False
    if len(ids) > 0 and ids[0] == start_idx:
        ids = ids[1:]
    for a_id in ids:
        if a_id == end_idx:
            break
        unpack.append(a_id)

    return unpack

def get_action(ids, start_idx, end_idx, id2word):
    action = []
    if len(ids) == 0:
        return ''
    if ids[0] == start_idx:
        ids = ids[1:]
    for a_id in ids:
        if a_id == end_idx:
            break
        action.append(id2word[a_id])
    return ' '.join(action)
            
def get_action_RL(ids, start_idx, end_idx, id2word):
    assert ids[0] == start_idx
    ids = ids[1:]
    ids = ids[:ids.index(end_idx)] if end_idx in ids else ids
    action = []
    ids_lst = [[]]
    for tid in ids:
        ids_lst[-1].append(tid)
        if id2word[tid] == ')':
            ids_lst.append([])
    ids_lst = ids_lst[:-1] if len(ids_lst[-1]) == 0 else ids_lst
    is_valid = True
    for i, lst in enumerate(ids_lst):
        temp_lst = [id2word[aid] for aid in lst]
        if  len(lst) < 3 or len(lst) > 7:
            is_valid = False

        elif len(lst) > 0 and not id2word[lst[0]].startswith('A'):
            is_valid = False
        
        elif 0 in lst or start_idx in lst:  # [PAD]
            is_valid = False
        
        elif temp_lst.count('(') != 1 or temp_lst.index('(') != 1:
            is_valid = False
        
        elif temp_lst.count(')') != 1 or temp_lst.index(')') != len(temp_lst) - 1:
            is_valid = False

        elif len(lst) > 0 and id2word[lst[0]].startswith('A'):
            if temp_lst[0] in ['A4', 'A5']:  # argmin, argmax
                is_valid &= len(temp_lst) == 3
            elif temp_lst[0] in ['A1', 'A2', 'A8', 'A9', 'A10', 'A16']:  # select, select_all, union, inter, differ
                if len(temp_lst) == 6: # (e, r, t)
                    is_valid &= (temp_lst[2].startswith("ENTITY") or temp_lst[2].startswith("TYPE") ) and temp_lst[3].startswith("RELATION") and temp_lst[4].startswith("TYPE")
                elif len(temp_lst) == 7: # (e, -r, t)
                    is_valid &=  (temp_lst[2].startswith("ENTITY") or temp_lst[2].startswith("TYPE") ) and temp_lst[3] == '-' and temp_lst[4].startswith("RELATION") and temp_lst[5].startswith("TYPE")
                else:
                    is_valid = False
            elif temp_lst[0] in ['A3', 'A6', 'A7']:  # bool(e), GreaterThan(e), LessThan(e), 
                is_valid &= len(temp_lst) == 4 and temp_lst[2].startswith("ENTITY")
            elif temp_lst[0] in ['A12', 'A13', 'A14']:  # ATLEAST(N), ATMOST(N), EQUAL(N), 
                is_valid &= len(temp_lst) == 4 and temp_lst[2] == 'INT'
            elif temp_lst[0] == 'A11': # count(), count(e)
                is_valid &= len(temp_lst) == 3 or len(temp_lst) == 4
            elif temp_lst[0] == 'A15': # around(e), around(e,r,t)
                if len(temp_lst) == 4:
                    is_valid &= temp_lst[2].startswith("ENTITY") or temp_lst[2] == 'INT'
                elif len(temp_lst) == 6: # (e, r, t)
                    is_valid &= temp_lst[2].startswith("ENTITY") and temp_lst[3].startswith("RELATION") and temp_lst[4].startswith("TYPE")
                elif len(temp_lst) == 7: # (e, -r, t)
                    is_valid &= temp_lst[2].startswith("ENTITY") and temp_lst[3] == '-' and temp_lst[4].startswith("RELATION") and temp_lst[5].startswith("TYPE")
                else:
                    is_valid = False
            
         
        if len(lst) > 0:
            for a_id in lst: 
                action.append(id2word[a_id])
            if len(lst) > 2:
                if id2word[lst[2]] == 'INT' and id2word[lst[0]] not in ['A12', 'A13', 'A14', 'A15']:
                    is_valid = False
        
    if is_valid: # check duplicate 
        for i in range(2, len(ids_lst)):
            if duplicate(ids_lst[i], ids_lst[i-1]) and duplicate(ids_lst[i-1], ids_lst[i-2]):
                is_valid = False
                break
            
    if len(action) == 0:
        return False, action
    return is_valid, action

def train_with_target(device, model, optimizer, data, symbol_list, start_idx, end_idx, pad_idx, logger, tf_writer=None, save_path=None, \
        epoch_num=50, batch_size=32, do_train=True, do_eval=True, finetune_bert=False, id2name=None, symbol_type=None):
    """
    data: tuple of (dataset, ques_emb, vocab_emb), dataset is list of dict
            ques_emb and vocab_emb are list of 'Tensor'
    """
    train_data = data
    eval_data = None
    if do_eval:
        instance_num = len(data[0])
        train_num = int(instance_num * 0.9)
        eval_num = instance_num - train_num
        if not finetune_bert:
            train_data = (data[0][:train_num], data[1][:train_num], data[2][:train_num])
            eval_data = (data[0][train_num:], data[1][train_num:], data[2][train_num:])
        else:
            train_data = (data[0][:train_num], None, None)
            eval_data = (data[0][train_num:], None, None)
        logger.info('training examples number: %d, validating examples number: %d' % (train_num, eval_num))
    else:
        logger.info('training examples number: %d, no validation' % (len(data[0])))
    max_bleu_score = 0.
    last_loss = 100000.
    symbol_dict = {  symbol:idx for idx, symbol in enumerate(symbol_list)}
    type_num = None
    if symbol_type is not None and len(symbol_type) > 0:
        word2type = {  idx:t for idx, t in enumerate(symbol_type)}
        type_num = 3
    else:
        word2type = {  idx:0 for idx, _ in enumerate(symbol_list)}
        type_num = 1
    for idx, symbol in enumerate(['E', 'R', 'T', 'I']):
        word2type[len(symbol_dict) + idx] = type_num
        type_num += 1
    for epoch_idx in range(epoch_num):
        if do_train:
            model.train()
            saved = False
            total_loss = 0.
            batch_num = math.ceil(len(train_data[0]) / batch_size)
            for batch_idx in range(batch_num):
                # torch.autograd.set_detect_anomaly(True)
                start_pos = batch_idx * batch_size
                end_pos = min(start_pos + batch_size, len(train_data[0]))
                ques_emb_lst = []
                vocab_emb_lst = []
                if finetune_bert:
                    for idx in range(start_pos, end_pos):
                        vocab_dict = {}
                        for mask, vocab in train_data[0][idx]['mask'].items():
                            vocab_name = str(vocab)
                            if mask != 'INT':
                                vocab_name = id2name[vocab]
                            vocab_dict[mask] = vocab_name
                        vocab_lst = list(vocab_dict.keys())
                        random.shuffle(vocab_lst)
                        ques_emb, vocab_emb = model.bert_encode(train_data[0][idx]['question'], [vocab_dict[vocab_mask] for vocab_mask in vocab_lst])
                        ques_emb_lst.append(ques_emb)
                        vocab_emb_lst.append(vocab_emb)
                        train_data[0][idx]['vocab'] = vocab_lst
                        train_data[0][idx]['length'] = ques_emb.size(0)
                src, lengths, target, target_emb, cand_emb, vocab_size, words, type_list, _ = batch_data(device, train_data[0][start_pos:end_pos], 
                                                                                        ques_emb_lst if finetune_bert else train_data[1][start_pos:end_pos],
                                                                                        vocab_emb_lst if finetune_bert else train_data[2][start_pos:end_pos], 
                                                                                        symbol_list, 
                                                                                        symbol_dict, 
                                                                                        start_idx, 
                                                                                        end_idx, 
                                                                                        pad_idx,
                                                                                        model.decoder.embeddings,
                                                                                        model.decoder.type_embeddings,
                                                                                        word2type)
                optimizer.zero_grad()
                # probs: [n x n_token]
                probs, _ = model(src, lengths, target, target_emb, cand_emb, type_list, vocab_size, testing=False)

                # remove start token
                # [batch x target_num x (target_len-1)]
                target = target[:, 1:].contiguous().view(-1)
                loss = compute_loss(probs, target, ignore_index=pad_idx) # [n]
                loss = loss.sum() / (end_pos - start_pos)
                # with torch.autograd.detect_anomaly():
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                logger.info('[Epoch %d/%d, batch %d/%d] loss: %.4f' % (epoch_idx+1, epoch_num, batch_idx+1, batch_num, loss.item()))
                if tf_writer is not None:
                    tf_writer.add_scalar('loss', loss.item(), epoch_idx * batch_num + batch_idx + 1)
            total_loss /= batch_num
            logger.info('[Epoch %d/%d] Mean loss: %.4f' % (epoch_idx+1, epoch_num, total_loss))
            if tf_writer is not None:
                tf_writer.add_scalar('mean_loss', total_loss, epoch_idx + 1)
        if do_eval:
            model.eval()
            batch_num = math.ceil(len(eval_data[0]) / batch_size)
            count = 0
            smooth = SmoothingFunction()
            bleu_total = 0.
            for batch_idx in range(batch_num):
                start_pos = batch_idx * batch_size
                end_pos = min(start_pos + batch_size, len(eval_data[0]))
                ques_emb_lst = []
                vocab_emb_lst = []
                if finetune_bert:
                    for idx in range(start_pos, end_pos):
                        vocab_dict = {}
                        for mask, vocab in eval_data[0][idx]['mask'].items():
                            vocab_name = str(vocab)
                            if mask != 'INT':
                                vocab_name = id2name[vocab]
                            vocab_dict[mask] = vocab_name
                        vocab_lst = list(vocab_dict.keys())
                        random.shuffle(vocab_lst)
                        with torch.no_grad():
                            ques_emb, vocab_emb = model.bert_encode(eval_data[0][idx]['question'], [vocab_dict[vocab_mask] for vocab_mask in vocab_lst])
                        ques_emb_lst.append(ques_emb)
                        vocab_emb_lst.append(vocab_emb)
                        eval_data[0][idx]['vocab'] = vocab_lst
                        eval_data[0][idx]['length'] = ques_emb.size(0)
                src, lengths, target, target_emb, cand_emb, vocab_size, words, type_list, reverse_dict = batch_data(device, eval_data[0][start_pos:end_pos], 
                                                                                        ques_emb_lst if finetune_bert else eval_data[1][start_pos:end_pos],
                                                                                        vocab_emb_lst if finetune_bert else eval_data[2][start_pos:end_pos], 
                                                                                        symbol_list, 
                                                                                        symbol_dict, 
                                                                                        start_idx, 
                                                                                        end_idx, 
                                                                                        pad_idx,
                                                                                        model.decoder.embeddings,
                                                                                        model.decoder.type_embeddings,
                                                                                        word2type)
                target = target.cpu().tolist()
                # results: dict
                with torch.no_grad():
                    results = model(src, lengths, None, None, cand_emb, type_list, vocab_size, testing=True)
                preds = results['predictions']
                bleu_score = 0.
                for ex_id, (pred, tgt) in enumerate(zip(preds, target)):
                    pred = unpack_ids(pred.cpu().tolist(), start_idx, end_idx)
                    tgt = unpack_ids(tgt, start_idx, end_idx)
                    bleu_score += sentence_bleu([tgt], pred, smoothing_function=smooth.method1, weights=(0.5, 0.5))
                    if count < 32:
                        logger.info('[Example %d] targets: %s' % (count+1, ' '.join([words[ex_id][idx] for idx in tgt])))
                        logger.info('[Example %d] preds: %s' % (count+1, ' '.join([words[ex_id][idx] for idx in pred])))
                    count += 1
                bleu_total += bleu_score
                logger.info('[Epoch %d/%d, batch %d/%d] bleu_score: %.4f' % (epoch_idx+1, epoch_num, batch_idx+1, batch_num, bleu_score / len(target)))
            bleu_total /= eval_num
            logger.info('[Epoch %d/%d] Mean bleu score: %.4f' % (epoch_idx+1, epoch_num, bleu_total))
            if tf_writer is not None:
                tf_writer.add_scalar('bleu_eval', bleu_total, epoch_idx + 1)
            if bleu_total > max_bleu_score:
                temp = max_bleu_score
                max_bleu_score = bleu_total
                saved = True
        if do_train and  saved and  save_path is not None:
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            logger.info('max_bleu_score: %.4f -> %.4f, save new model checkpoints to [%s]' % (temp, max_bleu_score, save_path))
        else:
            logger.info('max_bleu_score: %.4f, current_bleu_score: %.4f, do not save' % (max_bleu_score, bleu_total))
        # if do_train:
        #     last_loss = total_loss

def get_action_texts(actions, mask):
    mask_dict = {1: 'x', 2: 'y', 3:'z'}
    action_texts = []
    for action in actions:
        action_list = action.split(" ")
        action_list_list = [[]]
        for act in action_list:
            action_list_list[-1].append(act)
            if act == ')':
                action_list_list.append([])
        action_list_list = action_list_list[:-1]
        action_text = []
        a2_count = 0
        for idx, action_list in enumerate(action_list_list):
            assert action_list[0].startswith('A')
            is_a2 = not np.sum([l[0] != 'A2' for l in action_list_list[idx:]])
            if is_a2:
                action_text.append('<a1>')
                a2_count += 1
            else:
                action_text.append('<%s>' %  action_list[0].lower())
            for act in action_list[1:]:
                if act in mask:
                    if is_a2 and action_text[-1] != '<a1>' and act.startswith('TYPE'):
                        action_text.append('<entity>')
                        action_text.append(mask_dict[a2_count])
                        action_text.append('</entity>')
                    else:
                        action_text.append('<%s>' %  re.sub(r'[0-9]', '', act).lower())
                        action_text.append(mask[act].lower())
                        action_text.append('</%s>' %  re.sub(r'[0-9]', '', act).lower())
            else:
                if act == '(':
                    continue
                elif act == ')':
                    if is_a2:
                        action_text.append('</a1>')
                    else:
                        action_text.append('</%s>' %  action_list[0].lower())
                else:
                    action_text.append('<%s>' %  act.lower())
        action_text = ' '.join(action_text)
        action_texts.append(action_text)
    return action_texts

def update_dataset(action_list, reward, qa_info):
    global FINETUNE_DATASET, UPDATE_FLAG
    if reward < 1:
        return 
    act_set = set(action_list)
    # ensure coverage
    for mask in qa_info['mask'].keys():
        if mask not in act_set:
            return
    mask = qa_info['mask_name']
    action_text = []
    last_act = None
    for act in action_list:
        if act in mask:
            action_text.append('<%s>' %  re.sub(r'[0-9]', '', act).lower())
            action_text.append(mask[act].lower())
            action_text.append('</%s>' %  re.sub(r'[0-9]', '', act).lower())
        else:
            if act == '(':
                continue
            elif act == ')':
                action_text.append('</%s>' %  last_act)
            elif act.lower().startswith('a'):
                action_text.append('<%s>' %  act.lower())
                last_act = act.lower()
            else:
                action_text.append('<%s>' %  act.lower())
    action_text = ' '.join(action_text)
    # shortest length
    if qa_info['question'] in FINETUNE_DATASET:
        length = len(FINETUNE_DATASET[qa_info['question']]['action'].split(' '))
        if len(action_list) < length:
            FINETUNE_DATASET[qa_info['question']]['action'] = ' '.join(action_list)
            FINETUNE_DATASET[qa_info['question']]['action_text'] = action_text
            UPDATE_FLAG.add(qa_info['question'])
    else:
        FINETUNE_DATASET[qa_info['question']] = {'action': ' '.join(action_list), 'action_text': action_text, 'mask_name': qa_info['mask_name']}
        UPDATE_FLAG.add(qa_info['question'])
     


def get_and_update_reward(action_list, reward_memory, qa_info, adaptive, web_url, logger):
    global MEMORY_USE_COUNT, MEMORY_NOT_USE_COUNT
    if qa_info['state'] not in reward_memory:
        reward_memory[qa_info['state']] = {}
    if ' '.join(action_list) not in reward_memory[qa_info['state']]:
        sample_reward = calc_True_Reward(action_list, qa_info, adaptive_flag=adaptive, url=web_url)
        reward_memory[qa_info['state']][' '.join(action_list)] = sample_reward
        MEMORY_NOT_USE_COUNT += 1
    else:
        sample_reward = reward_memory[qa_info['state']][' '.join(action_list)]
        MEMORY_USE_COUNT += 1
    if (MEMORY_NOT_USE_COUNT + MEMORY_USE_COUNT) % 1000 == 0:
        logger.info('Use reward memory %d times in past %d reward computation!' %  (MEMORY_USE_COUNT, MEMORY_NOT_USE_COUNT + MEMORY_USE_COUNT) )
    return sample_reward

def get_auxiliary_reward(action_list, vocab_dict, d_list, reward_memory, adaptive, web_url, logger):
    if len(d_list) == 0:
        return 0., 0.
    reward = 0.
    weight = 0.
    action_list = [ vocab_dict[act] if act in vocab_dict else act for act in action_list]
    for d in d_list:
        relation_values = [v for k, v in d['mask'].items() if k.startswith('RELATION') ]
        relation_masks = [k for k, v in d['mask'].items() if k.startswith('RELATION') ]
        reward_lst = []
        for relations in itertools.permutations(relation_masks):
            relation_dict = {k:v for k, v in zip(relations, relation_values)}
            new_action = [ relation_dict[act] if act in relation_dict else str(d['mask'][act]) if act in d['mask'] else act for act in action_list]
            reward_lst.append(get_and_update_reward(new_action, reward_memory, d, adaptive, web_url, logger))
        reward += d['weight'] * np.max(reward_lst)
        weight += d['weight'] 
    return reward, weight

def train_RL(device, model, optimizer, data, symbol_list, start_idx, end_idx, pad_idx, logger, tf_writer=None, save_path=None, \
                reward_save_path=None, memory_buffer_json=None, reward_memory=None, memory_buffer=None, adaptive=False, web_url=None, \
                epoch_num=50, batch_size=32, do_train=True, do_eval=True, finetune_bert=False, id2name=None, symbol_type=None, \
                reward_lambda=0.5, loaded_model_path=None):
    """
    data: tuple of (dataset, ques_emb, vocab_emb), dataset is list of dict
            ques_emb and vocab_emb are list of 'Tensor'
    """
    global FINETUNE_DATASET, UPDATE_FLAG
    train_data = data
    if do_eval:
        instance_num = len(data[0])
        train_num = int(instance_num * 0.85)
        eval_num = instance_num - train_num
        if not finetune_bert:
            train_data = (data[0][:train_num], data[1][:train_num], data[2][:train_num])
            eval_data = (data[0][train_num:], data[1][train_num:], data[2][train_num:])
        else:
            train_data = (data[0][:train_num], None, None)
            eval_data = (data[0][train_num:], None, None)
        train_num = len(train_data[0])
        eval_num = len(eval_data[0])
        logger.info('training examples number: %d, validating examples number: %d' % (train_num, eval_num))
    else:
        logger.info('training examples number: %d, no validation' % (len(data[0])))
    max_eval_reward = 0.
    symbol_dict = {  symbol:idx for idx, symbol in enumerate(symbol_list)}
    type_num = None
    if symbol_type is not None and len(symbol_type) > 0:
        word2type = {  idx:t for idx, t in enumerate(symbol_type)}
        type_num = 3
    else:
        word2type = {  idx:0 for idx, _ in enumerate(symbol_list)}
        type_num = 1
    for idx, symbol in enumerate(['E', 'R', 'T', 'I']):
        word2type[len(symbol_dict) + idx] = type_num
        type_num += 1
    batch_loss_policy = 0.
    for epoch_idx in range(epoch_num):
        if do_train:
            model.train()
            saved = False
            total_loss = 0.
            true_reward_argmax = []
            true_reward_sample = []
            batch_num = math.ceil(len(train_data[0]) / batch_size)
            lambda_value = min(1.0, float(math.pow(1.0 + ETA, float(epoch_idx) + 1.0) * LAMBDA_0))
            for batch_idx in range(batch_num):
                start_pos = batch_idx * batch_size
                end_pos = min(start_pos + batch_size, len(train_data[0]))
                ques_emb_lst = []
                vocab_emb_lst = []
                if finetune_bert:
                    for idx in range(start_pos, end_pos):
                        vocab_dict = {}
                        for mask, vocab in train_data[0][idx]['mask'].items():
                            vocab_name = str(vocab)
                            if mask != 'INT':
                                vocab_name = id2name[vocab]
                            vocab_dict[mask] = vocab_name
                        vocab_lst = list(vocab_dict.keys())
                        random.shuffle(vocab_lst)
                        ques_emb, vocab_emb = model.bert_encode(train_data[0][idx]['question'], [vocab_dict[vocab_mask] for vocab_mask in vocab_lst])
                        ques_emb_lst.append(ques_emb)
                        vocab_emb_lst.append(vocab_emb)
                        train_data[0][idx]['vocab'] = vocab_lst
                        train_data[0][idx]['length'] = ques_emb.size(0)
                src, lengths, target, target_emb, cand_emb, vocab_size, words, type_list, reverse_dict = batch_data(device, train_data[0][start_pos:end_pos], 
                                                                                        ques_emb_lst if finetune_bert else train_data[1][start_pos:end_pos],
                                                                                        vocab_emb_lst if finetune_bert else train_data[2][start_pos:end_pos], 
                                                                                        symbol_list, 
                                                                                        symbol_dict, 
                                                                                        start_idx, 
                                                                                        end_idx, 
                                                                                        pad_idx,
                                                                                        model.decoder.embeddings,
                                                                                        model.decoder.type_embeddings,
                                                                                        word2type)
                
                optimizer.zero_grad()
                argmax_results, beam_results = model(src, lengths=lengths, vocab_emb=cand_emb, vocab_type=type_list, vocab_size=vocab_size, testing=False)
                net_losses = []
                for ex_id in range(src.size(0)):
                    offset_idx = reverse_dict[ex_id]
                    qa_info = train_data[0][start_pos+ex_id]
                    is_valid, action_list = get_action_RL(argmax_results['predictions'][offset_idx].cpu().tolist(), start_idx, end_idx, words[offset_idx])
                    new_action = [ qa_info['mask'][act] if act in qa_info['mask'] else act for act in action_list]
                    argmax_reward = get_and_update_reward(new_action, reward_memory, qa_info, adaptive, web_url, logger) if is_valid else -1
                    temp = argmax_reward
                    if is_valid and reward_lambda > 0:
                        bias_reward, bias_weight = get_auxiliary_reward(action_list, qa_info['vocab_dict'], qa_info['sim_d'], reward_memory, adaptive, web_url, logger)
                        if bias_weight > 0:
                            argmax_reward = (1.0 * argmax_reward + reward_lambda * bias_reward) / (1.0 + reward_lambda * bias_weight)
                    true_reward_argmax.append(argmax_reward)
                    if batch_idx < 1:
                        logger.info("%s" % qa_info['state'])
                        logger.info("Input: %s" % qa_info['question'])
                        logger.info("orig_response: %s" %  qa_info['orig_response'])
                        logger.info("Argmax: %s, reward_nobias=%.4f, reward=%.4f, valid:%s" % (' '.join(action_list), temp, argmax_reward, str(is_valid)))

                    qid = qa_info['state']
                    action_memory = []
                    inner_net_policies = []
                    # inner_net_actions = []
                    inner_net_advantages = []
                    for sample_score, sample_pred, sample_prob in zip(beam_results["scores"][offset_idx], beam_results["predictions"][offset_idx], beam_results["probs"][offset_idx]):
                        is_valid, action_list = get_action_RL(sample_pred.cpu().tolist(), start_idx, end_idx, words[offset_idx])
                        new_action = [ qa_info['mask'][act] if act in qa_info['mask'] else act for act in action_list]
                        sample_reward = get_and_update_reward(new_action, reward_memory, qa_info, adaptive, web_url, logger) if is_valid else -1
                        update_dataset(action_list, sample_reward, qa_info)
                        temp = sample_reward
                        if is_valid and reward_lambda > 0:
                            bias_reward, bias_weight = get_auxiliary_reward(action_list, qa_info['vocab_dict'], qa_info['sim_d'], reward_memory, adaptive, web_url, logger)
                            if bias_weight > 0:
                                sample_reward = (1.0 * sample_reward + reward_lambda * bias_reward) / (1.0 + reward_lambda * bias_weight)
                        true_reward_sample.append(sample_reward)
                        if batch_idx < 1:
                            logger.info("Sample: %s, reward_nobias=%.4f, reward=%.4f, score:%.4f, valid: %s" % (' '.join(action_list), temp, sample_reward, sample_score, str(is_valid)))
                        if sample_reward >= argmax_reward and sample_reward > 0.0:
                            action_memory.append(new_action)
                        action_buffer = memory_buffer[qid] if qid in memory_buffer else None
                        F_proximity = calculate_proximity(new_action, action_buffer)
                        F_diversity = calculate_diversity(new_action, action_buffer)
                        reward_bonus = lambda_value * F_proximity + (1.0 - lambda_value) * F_diversity
                        # Using scalar α to scale the bonus.
                        regularized_reward_bonus = ALPHA * reward_bonus
                        inner_net_policies.append(sample_prob)
                        # inner_net_actions.append(sample_pred)
                        advantages = [sample_reward - argmax_reward + regularized_reward_bonus] * len(sample_pred)
                        inner_net_advantages.extend(advantages)
                    
                    if len(action_memory) > 0:
                        if qid not in memory_buffer:
                            memory_buffer[qid] = action_memory
                        else:
                            q_memory = memory_buffer[qid]
                            for action_tokens in action_memory:
                                duplicate_flag = False
                                if len(q_memory) > 0:
                                    for temp_list in q_memory:
                                        if duplicate(temp_list, action_tokens):
                                            duplicate_flag = True
                                            break
                                    if not duplicate_flag:
                                        # If buffer is full, remove one element randomly.
                                        if len(q_memory) == MAX_MEMORY_BUFFER_SIZE:
                                            random_index = randrange(0, len(q_memory))
                                            q_memory.pop(random_index)
                                        q_memory.append(action_tokens)
                                        memory_buffer[qid] = q_memory
                                else:
                                    q_memory.append(action_tokens)
                                    memory_buffer[qid] = q_memory

                    inner_log_prob_v = torch.cat(inner_net_policies).log() # [sample_num*seq_len]
                    # inner_actions_t = torch.cat(inner_net_actions) #  # [sample_num*seq_len]
                    inner_adv_v = torch.FloatTensor(inner_net_advantages).to(device)  # [sample_num*seq_len]
                    inner_log_prob_actions_v = inner_adv_v * inner_log_prob_v
                    inner_loss_policy_v = -inner_log_prob_actions_v.mean()
                    if inner_log_prob_v.isnan().any() or inner_log_prob_v.isinf().any():
                        logger.error('inner_log_prob_v, ex_id:%d, nan' % ex_id)
                        logger.error(torch.cat(inner_net_policies))
                        # for param_name, param in model.named_parameters():
                        #     print(param_name, param.isnan().any())
                        print(sum([param.isnan().any() for param_name, param in model.named_parameters()]))
                    if inner_adv_v.isnan().any() or inner_adv_v.isinf().any():
                        logger.error('inner_adv_v, ex_id:%d, nan' % ex_id)    
                        # for param_name, param in model.named_parameters():
                        #     logger.error('%s: %s' % (param_name, str(param.isnan().any())))
                        logger.error(sum([param.isnan().any() for param_name, param in model.named_parameters()]))
                    # Record the loss for each task in a batch.
                    net_losses.append(inner_loss_policy_v)
         
                loss = torch.stack(net_losses).mean()
                assert torch.isnan(loss).sum() == 0, loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                logger.info('[Epoch %d/%d, batch %d/%d] loss: %.4f' % (epoch_idx+1, epoch_num, batch_idx+1, batch_num, loss.item()))
                batch_loss_policy += loss.item()
                if tf_writer is not None:
                    temp =  epoch_idx * batch_num + batch_idx + 1
                    if temp % 100 == 0:
                        tf_writer.add_scalar('loss_policy', batch_loss_policy / 100., temp)
                        batch_loss_policy = 0.
                    # tf_writer.add_scalar('loss_policy', loss.item(), epoch_idx * batch_num + batch_idx + 1)
            total_loss /= batch_num
            logger.info('[Epoch %d/%d] Mean loss: %.4f' % (epoch_idx+1, epoch_num, total_loss))
            if tf_writer is not None:
                tf_writer.add_scalar('mean_policy_loss', total_loss, epoch_idx + 1)
                tf_writer.add_scalar('reward_armax', np.mean(true_reward_argmax), epoch_idx + 1)
                tf_writer.add_scalar('reward_sample', np.mean(true_reward_sample), epoch_idx + 1)

            total_loss /= batch_num

        if do_eval:
            if loaded_model_path is not None:
                model_name = [ filename  for filename in os.listdir(loaded_model_path)  if  filename.startswith('epoch_%d_' % (epoch_idx+1)) ][0]
                model_dic = torch.load(os.path.join(loaded_model_path, model_name), map_location=device)
                model.load_state_dict(model_dic)
                logger.info("successfully load pre-trained model [%s]" % model_name)

            reward_dict = {}
            question_num_dict = {}
            model.eval()
            batch_num = math.ceil(len(eval_data[0]) / batch_size)
            count = 0
            reward_total = 0.
            for batch_idx in range(batch_num):
                start_pos = batch_idx * batch_size
                end_pos = min(start_pos + batch_size, len(eval_data[0]))
                ques_emb_lst = []
                vocab_emb_lst = []
                if finetune_bert:
                    for idx in range(start_pos, end_pos):
                        vocab_dict = {}
                        for mask, vocab in eval_data[0][idx]['mask'].items():
                            vocab_name = str(vocab)
                            if mask != 'INT':
                                vocab_name = id2name[vocab]
                            vocab_dict[mask] = vocab_name
                        vocab_lst = list(vocab_dict.keys())
                        random.shuffle(vocab_lst)
                        with torch.no_grad():
                            ques_emb, vocab_emb = model.bert_encode(eval_data[0][idx]['question'], [vocab_dict[vocab_mask] for vocab_mask in vocab_lst])
                        ques_emb_lst.append(ques_emb)
                        vocab_emb_lst.append(vocab_emb)
                        eval_data[0][idx]['vocab'] = vocab_lst
                        eval_data[0][idx]['length'] = ques_emb.size(0)
                src, lengths, _, _, cand_emb, vocab_size, words, type_list, reverse_dict = batch_data(device, eval_data[0][start_pos:end_pos], 
                                                                                        eval_data[1][start_pos:end_pos],
                                                                                        eval_data[2][start_pos:end_pos], 
                                                                                        symbol_list, 
                                                                                        symbol_dict, 
                                                                                        start_idx, 
                                                                                        end_idx, 
                                                                                        pad_idx,
                                                                                        model.decoder.embeddings,
                                                                                        model.decoder.type_embeddings,
                                                                                        word2type)
    
                # results: dict
                with torch.no_grad():
                    results = model(src, lengths, None, None, cand_emb, type_list, vocab_size, testing=True)
                preds = results['predictions']
                reward_score = 0.
                for ex_id in range(len(preds)):
                    offset_idx = reverse_dict[ex_id]
                    qa_info = eval_data[0][start_pos+ex_id]
                    is_valid, pred_action = get_action_RL(preds[offset_idx].cpu().tolist(), start_idx, end_idx, words[offset_idx])
                    new_action = [ qa_info['mask'][act] if act in qa_info['mask'] else act for act in pred_action]
                    reward = get_and_update_reward(new_action, reward_memory, qa_info, adaptive, web_url, logger) if is_valid else -1
                    temp = reward
                    # reward_lambda = 1.0
                    if is_valid  and reward_lambda > 0:
                        bias_reward, bias_weight = get_auxiliary_reward(pred_action, qa_info['vocab_dict'], qa_info['sim_d'], reward_memory, adaptive, web_url, logger)
                        if bias_weight > 0:
                            reward = (1.0 * reward + bias_reward) / (1.0 + bias_weight)
                            # reward = (1.0 * reward + reward_lambda * bias_reward) / (1.0 + reward_lambda * bias_weight)
                    if count < 200:
                        # logger.info('[Example %d] qid: %s preds: %s, reward: %.4f, valid: %s' % (count+1, qa_info['state'], ' '.join(pred_action), reward, str(is_valid)))
                        logger.info('[Example %d] qid: %s preds: %s, reward: %.4f, bias_reward: %.4f, valid: %s' % (count+1, qa_info['state'], ' '.join(pred_action), temp, reward, str(is_valid)))
                    count += 1
                    reward_score += reward
                    question_type = re.sub(r'[0-9]+', '', qa_info['state'])
                    question_type = question_type.replace('_', '').replace(' ', '')
                    if question_type not in reward_dict:
                        reward_dict[question_type] = 0
                        question_num_dict[question_type] = 0
                    reward_dict[question_type] += reward
                    question_num_dict[question_type] += 1
                    # if question_type == 'QuantitativeReasoning(Count)(All)':
                    #     logger.info('%s: %.4f(%.4f)' % (qa_info['state'], reward, float(reward_dict[question_type])/float(question_num_dict[question_type])))


                logger.info('[Epoch %d/%d, batch %d/%d] mean reward: %.4f' % (epoch_idx+1, epoch_num, batch_idx+1, batch_num, reward_score / len(preds)))
                reward_total += reward_score
            reward_total /= eval_num
            for k in reward_dict.keys():
                reward_dict[k] = float(reward_dict[k]) / float(question_num_dict[k])
            logger.info('[Epoch %d/%d] Mean reward: %.4f' % (epoch_idx+1, epoch_num, reward_total))
            for k, v in reward_dict.items():
                logger.info('[Epoch %d/%d] reward_eval_%s : %.4f' % (epoch_idx+1, epoch_num, k, v))
            if tf_writer is not None:
                tf_writer.add_scalar('reward_eval', reward_total, epoch_idx + 1)
                for k, v in reward_dict.items():
                    k =  re.sub('\(|\)', '_', k)
                    tf_writer.add_scalar('reward_eval_%s' % k, v, epoch_idx + 1)
            if reward_total > max_eval_reward:
                temp = max_eval_reward
                max_eval_reward = reward_total
                saved = True
        if do_train:
            model_save_path = os.path.join(save_path, 'epoch_%d_reward_%.4f_%.4f.bin' % (epoch_idx+1, np.mean(true_reward_argmax), reward_total))
            torch.save(model.state_dict(), model_save_path)
            logger.info('reward_score: %.4f, max_reward_score: %.4f, save new model checkpoints to [%s]' % (reward_total, max_eval_reward, model_save_path))
        if reward_save_path is not None:
            json.dump(reward_memory, open(reward_save_path, 'w'), ensure_ascii=False)
            # with open(reward_save_path, 'w') as f:
            #     for action, reward in reward_memory.items():
            #         f.write('%s\t%f\n' % (action, reward))
            logger.info('update reward memory')
            # logger.info('update reward memory with size [%d -> %d]' % (reward_memory_length, len(reward_memory) ))
            # reward_memory_length = len(reward_memory)

        json.dump(memory_buffer, open(memory_buffer_json, 'w'))
        logger.info('update action memory buffer to [%s]' % (memory_buffer_json))
        

def predict(device, model, data, symbol_list, start_idx, end_idx, pad_idx, logger, output_path, batch_size=1, finetune_bert=False, id2name=None, beam_search=False, top_k=5, reward_memory=None, web_url=None):
    """
    data: tuple of (dataset, ques_emb, vocab_emb), dataset is list of dict
            ques_emb and vocab_emb are list of 'Tensor'
    """

    logger.info('testing examples number: %d' % (len(data[0])))
    symbol_dict = {  symbol:idx for idx, symbol in enumerate(symbol_list)}
    model.eval()
    batch_num = math.ceil(len(data[0]) / batch_size)
    word2type = {  idx:0 for idx, _ in enumerate(symbol_list)}
    type_num = 1
    for idx, symbol in enumerate(['E', 'R', 'T', 'I']):
        word2type[len(symbol_dict) + idx] = type_num
        type_num += 1
    for batch_idx in tqdm(range(batch_num)):
        start_pos = batch_idx * batch_size
        end_pos = min(start_pos + batch_size, len(data[0]))
        ques_emb_lst = []
        vocab_emb_lst = []
        if finetune_bert:
            for idx in range(start_pos, end_pos):
                vocab_dict = {}
                for mask, vocab in data[0][idx]['mask'].items():
                    vocab_name = str(vocab)
                    if mask != 'INT':
                        vocab_name = id2name[vocab]
                    vocab_dict[mask] = vocab_name
                vocab_lst = list(vocab_dict.keys())
                random.shuffle(vocab_lst)
                with torch.no_grad():
                    ques_emb, vocab_emb = model.bert_encode(data[0][idx]['question'], [vocab_dict[vocab_mask] for vocab_mask in vocab_lst])
                ques_emb_lst.append(ques_emb)
                vocab_emb_lst.append(vocab_emb)
                data[0][idx]['vocab'] = vocab_lst
                data[0][idx]['length'] = ques_emb.size(0)
        src, lengths, _, _, cand_emb, vocab_size, words, type_list, reverse_dict = batch_data(device, data[0][start_pos:end_pos], 
                                                                                        ques_emb_lst if finetune_bert else data[1][start_pos:end_pos],
                                                                                        vocab_emb_lst if finetune_bert else data[2][start_pos:end_pos], 
                                                                                        symbol_list, 
                                                                                        symbol_dict, 
                                                                                        start_idx, 
                                                                                        end_idx, 
                                                                                        pad_idx,
                                                                                        model.decoder.embeddings,
                                                                                        model.decoder.type_embeddings,
                                                                                        word2type)
        # results: dict
        with torch.no_grad():
            if beam_search:
                results = model(src, lengths=lengths, vocab_emb=cand_emb, vocab_type=type_list, vocab_size=vocab_size, testing=True, top_k=top_k, beam_search=True)
            else:
                results = model(src, lengths, None, None, cand_emb, type_list, vocab_size, testing=True)
        
        for ex_id in range(len(results['predictions'])):
            offset_idx = reverse_dict[ex_id]
            qa_info = data[0][start_pos+ex_id]
            pred = results['predictions'][offset_idx]
            if beam_search:
                if batch_idx < 1:
                    logger.info('[Example %d] %s' % (batch_idx * batch_size+ex_id+1, qa_info['question']))
                max_prob = 0.
                action = None
                for act_idx, act in enumerate(pred):
                    prob = np.exp(results['scores'][offset_idx][act_idx].item())
                    biaes_prob = 0.
                    is_valid, action_list = get_action_RL(act.cpu().tolist(), start_idx, end_idx, words[offset_idx])
                    if is_valid:
                        bias_reward, bias_weight = get_auxiliary_reward(action_list, qa_info['vocab_dict'], qa_info['sim_d'], reward_memory, True, web_url, logger)
                        if bias_weight > 0:
                            # biaes_prob = (1.0 * prob + 0.5 * bias_reward) / (1.0 + 0.5 * bias_weight)
                            biaes_prob = bias_reward / (1.0 * bias_weight)
                    if batch_idx < 1:
                        logger.info('action_list: %s, valid: %s, prob: %.4f, bais_prob: %.4f' % (' '.join(action_list), str(is_valid), prob, biaes_prob))
                    if biaes_prob > max_prob:
                        max_prob = biaes_prob
                        action = action_list
                if action == None:
                    action = get_action_RL(pred[0].cpu().tolist(), start_idx, end_idx, words[offset_idx])[1]
                action = ' '.join(action)
                # logger.info('action_list: %s, max_prob:%.4f' % (action, max_prob))
                        
            else:
                action = get_action(pred.cpu().tolist(), start_idx, end_idx, words[offset_idx])
            # if action[-1] != ')':
            #     print(action)
            output_path.write(action + '\n')
            output_path.flush()



        






