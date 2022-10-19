import os

import sys
import torch
import numpy as np
import random
import json
from tqdm import tqdm
from pytorch_pretrained_bert import tokenization, BertModel


def get_model_and_tokenizer_bert(model_name):
    bert = BertModel.from_pretrained(model_name)
    tokenizer = tokenization.BertTokenizer.from_pretrained(model_name)
    bert.eval()
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU!")
    else:
        device = "cpu"
        print("GPU not available.")
    bert.to(device)
    return bert, tokenizer

def prepare_sentence_for_bert(sent, vocab, tokenizer):
    tokens = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
    start_idx = len(tokens)
    tokens +=  tokenizer.tokenize(vocab) + ['[SEP]']
    end_idx = len(tokens) - 1
    return tokens, (start_idx, end_idx)

def get_attention(sequence):
    attention = [1 for i in sequence if i != 0]
    attention.extend([0 for i in sequence if i ==0])
    return attention

def get_batch_results_from_bert(tokens_list, tokenizer, bert):
    max_seq_size = max([len(tokens) for tokens in tokens_list])
    input_id_list = []
    attention_list = []
    for sequence_tokens in tokens_list:
        sequence_ids = tokenizer.convert_tokens_to_ids(sequence_tokens)
        sequence_ids.extend([0] * (max_seq_size - len(sequence_ids)))
        input_id_list.append(sequence_ids)
        attention_list.append(get_attention(sequence_ids))
    input_tensor = torch.LongTensor(input_id_list)
    attention_tensor = torch.LongTensor(attention_list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = input_tensor.to(device)
    attention_tensor = attention_tensor.to(device)
    with torch.no_grad():
        bert_output = bert(input_ids=input_tensor, attention_mask=attention_tensor, output_all_encoded_layers=False) 
    return bert_output

if __name__ == '__main__':
    model, tokenizer = get_model_and_tokenizer_bert('bert-base-uncased')
    print('bert model loaded!')
    for data_dir in ['pretrain', 'rl', 'test/test_sample']:    
        print('start to encode [%s]' % data_dir)
        # load pretrain_dataset
        dataset = []
        for line in open(os.path.join(data_dir, 'data.json')):
            dataset.append(json.loads(line)) 
        print('dataset loaded!')
        embedding_dir = os.path.join(data_dir, 'embedding')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
        f_ques = open(os.path.join(embedding_dir, 'question.bin'), 'wb')
        f_vocab = open(os.path.join(embedding_dir, 'vocab.bin'), 'wb')
        f_vocab_list = open(os.path.join(embedding_dir, 'vocab.txt'), 'w')
        for instance in tqdm(dataset):
            tokens_list = []
            tokens_list.append(['[CLS]'] + tokenizer.tokenize(instance['question']) + ['[SEP]'])
            ques_len = len(tokens_list[0])
            target_idx_list = []
            vocab_list = []
            for mask, vocab in instance['mask_name'].items():
                vocab_name = str(vocab)
                # if mask != 'INT':
                #     vocab_name = id2name[vocab]
                tokens, target_idx = prepare_sentence_for_bert(instance['question'], vocab_name, tokenizer)
                tokens_list.append(tokens)
                target_idx_list.append(target_idx)
                vocab_list.append(mask)
            # ([batch_size, sequence_length, hidden_size], [batch_size, hidden_size])
            batch_model_results = get_batch_results_from_bert(tokens_list, tokenizer, model)
            # sent emb
            question_emb = batch_model_results[0][0][1:ques_len-1]
            np.save(f_ques, question_emb.cpu().numpy())
            # vocab_emb 
            hidden_size = batch_model_results[0].shape[-1]
            vocab_emb = torch.zeros((len(vocab_list), hidden_size), dtype=question_emb.dtype)
            for i, (start, end) in enumerate(target_idx_list):
                vocab_emb[i, :] = batch_model_results[0][i+1, start:end, :].mean(dim=0).cpu()
            np.save(f_vocab, vocab_emb.numpy()) 
            f_vocab_list.write('|'.join(vocab_list) + '\n')
        
        f_ques.close()
        f_vocab.close()
        f_vocab_list.close()


    