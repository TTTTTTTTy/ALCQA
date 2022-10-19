import os
import warnings
warnings.filterwarnings('ignore')
import sys
import torch
import re
import math
import json
import random
from tqdm import tqdm
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import argparse
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Name of trained question rewriting model")

    args = parser.parse_args()
    logger = init_logger()
    model_load_path = os.path.join('saved_models/question_decompose', args.model_name)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base/')
    added_tokens = ['<entity>', '</entity>', '<relation>', '</relation>', '<type>', '</type>', '<int>', '</int>']
    logger.info('special tokens: %s' % ' '.join(added_tokens))
    tokenizer.add_tokens(added_tokens)
    config = BartConfig.from_pretrained('facebook/bart-base')
    config.forced_bos_token_id = None
    model = BartForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model_dic = torch.load(model_load_path, map_location='cpu')
    model.load_state_dict(model_dic)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    for data_dir in ['pretrain', 'rl', 'test/test_sample']:  
        data_path = 'data/%s/data.json' % data_dir
        out = open('data/%s/decomposed_predict.txt' % data_dir, 'w')
        logger.info('pretrained model loaded')

        for line in tqdm(open(data_path).readlines()):
            d = json.loads(line)
            input_text = [d['question']]
            for mask, m_name in d['mask_name'].items():
                input_text.append('<%s>' % re.sub(r'[0-9]', '', mask.lower()))
                input_text.append(m_name)
                input_text.append('</%s>' % re.sub(r'[0-9]', '', mask.lower()))
            input_text = ' '.join(input_text)
            inputs = tokenizer([input_text], padding=True,return_tensors='pt')
            summary_ids = model.generate(inputs['input_ids'].to(device), num_beams=3, min_length=5, max_length=100)
            out.write(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False) + '\n')
            out.flush()
