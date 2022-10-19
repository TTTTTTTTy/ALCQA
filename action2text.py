import os
import warnings
warnings.filterwarnings('ignore')
import torch
import re
import math
import json
import random
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.optimization import AdamW, get_scheduler
from sacrebleu import compute_bleu, corpus_bleu as _corpus_bleu

from utils import init_logger

def sentence_bleu(hypothesis, reference):
    bleu = _corpus_bleu(hypothesis, reference)
    for i in range(1, 4):
        bleu.counts[i] += 1
        bleu.totals[i] += 1
    bleu = compute_bleu(
        bleu.counts,
        bleu.totals,
        bleu.sys_len,
        bleu.ref_len,
        smooth_method="exp",
    )
    return bleu.score

logger = init_logger()

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
added_tokens = ['<entity>', '</entity>', '<relation>', '</relation>', '<type>', '</type>', '<int>', '</int>', '<->', '<&>', '<|>']
for i in range(16):
    added_tokens.append('<a%d>' % (i+1))
    added_tokens.append('</a%d>' % (i+1))
logger.info('special tokens: %s' % ' '.join(added_tokens))
tokenizer.add_tokens(added_tokens)
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model.resize_token_embeddings(len(tokenizer))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
logger.info('pretrained model loaded')

data_path = 'data/pretrain/data.json'
dataset = []
for line in open(data_path).readlines():
    d = json.loads(line)
    action_list = d['action'].split(" ")
    mask = d['mask_name']
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
    dataset.append((action_text, d['question'].lower()))

logger.info('Dataset: %d examples' % len(dataset))
train_dataset = dataset[:-500]
dev_dataset = dataset[-500:]
logger.info('training examples: %d, dev examples: %d' % (len(train_dataset), len(dev_dataset)))

batch_size = 32
epoch_num = 30

optimizer = AdamW(model.parameters(), lr=1e-5)

max_blue_score = 0.
for epoch_idx in range(epoch_num):
    # train
    batch_num =  math.ceil(len(train_dataset) / batch_size)
    random.shuffle(train_dataset)
    model.train()
    total_loss = 0.
    for batch_idx in range(batch_num):
        start_pos = batch_idx * batch_size
        end_pos = min(start_pos + batch_size, len(dataset))
        batch = train_dataset[start_pos: end_pos]
        inputs = tokenizer([x[0] for x in batch], padding=True,return_tensors='pt')
        outputs = tokenizer([x[1] for x in batch], padding=True, return_tensors='pt')
        decoder_input_ids = outputs['input_ids'][:, :-1]
        optimizer.zero_grad()
        labels = torch.where(outputs['input_ids'][:, 1:] == tokenizer.pad_token_id, torch.full_like(outputs['input_ids'][:, 1:], -100), outputs['input_ids'][:, 1:])
        loss = model(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device), decoder_input_ids=decoder_input_ids.to(device), labels=labels.to(device))[0]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            logger.info('[Epoch %d/%d, batch %d/%d] loss: %.4f' % (epoch_idx+1, epoch_num, batch_idx+1, batch_num, loss.item()))
    logger.info('[Epoch %d/%d] mean loss: %.4f' % (epoch_idx+1, epoch_num, total_loss / batch_num))
    
    # dev 
    dev_batch_num = math.ceil(len(dev_dataset) / batch_size)     
    mean_bleu_score = 0.
    model.eval()
    for batch_idx in range(dev_batch_num):       
        start_pos = batch_idx * batch_size
        end_pos = min(start_pos + batch_size, len(dev_dataset))
        batch = dev_dataset[start_pos: end_pos]         
        inputs = tokenizer([x[0] for x in batch], padding=True,return_tensors='pt')
        targets = [ x[1] for x in batch]
        with torch.no_grad():
            predict_ids = model.generate(inputs['input_ids'].to(device), num_beams=3, min_length=5, max_length=30)
        prdict_texts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in predict_ids]
        bleu_scores = [sentence_bleu(tgt, pred) for pred, tgt in zip(prdict_texts, targets)]
        for example, pred in zip(batch, prdict_texts):
            logger.info('action: %s' % example[0])
            logger.info('target: %s' % example[1])
            logger.info('predict: %s' % pred)
        if (batch_idx + 1) % 10 == 0:
            logger.info('[Epoch %d/%d, batch %d/%d] bleu score: %.4f' % (epoch_idx+1, epoch_num, batch_idx+1, batch_num, np.mean(bleu_scores)))
        mean_bleu_score += np.sum(bleu_scores)
    mean_bleu_score /= len(dev_dataset)
    if mean_bleu_score > max_blue_score:
        max_blue_score = mean_bleu_score
    logger.info('[Epoch %d/%d] mean bleu score: %.4f, max bleu score: %.4f' % (epoch_idx+1, epoch_num, mean_bleu_score, max_blue_score))
    model_save_path = os.path.join('saved_models/action2text/', 'epoch_%d_score_%.4f.bin' % (epoch_idx+1, mean_bleu_score))
    torch.save(model.state_dict(), model_save_path)
    logger.info('save new model checkpoints to [%s]' % (model_save_path))