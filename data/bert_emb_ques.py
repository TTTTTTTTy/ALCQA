from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel
import torch
import os
import numpy as np
from tqdm import tqdm
import json

device = torch.device('cuda:0')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()
for data_dir in ['pretrain', 'rl', 'test/test_sample']:    
    f_ques = open(os.path.join('%s/embedding' % data_dir, 'decomposed_predict.bin'), 'wb')
    count = 0
    lines = open('%s/decomposed_predict.txt' % data_dir).readlines()
    lines1 =  open('%s/data.json' % data_dir).readlines()
    assert len(lines) == len(lines1)
    for line, lines1 in tqdm(zip(lines, lines1), total=len(lines)):
        d = json.loads(lines1)
        line = d['question'].lower() + ' [SEP] ' + line.strip()
        # line = line.strip()
        tokens = bert_tokenizer.encode(line)
        with torch.no_grad():
            result = bert_model(input_ids=torch.tensor([tokens]).to(device), output_all_encoded_layers=False)[0][0]
        np.save(f_ques, result[1:-1].cpu().numpy())
        count += 1
    print(count)