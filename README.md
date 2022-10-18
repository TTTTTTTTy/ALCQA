# ALCQA
Code of EMNLP 2022 Paper "Improving Complex Knowledge Base Answering via Question-to-Action and Question-to-Question Alignment"

## Dependencies:
- torch
- transformers
- pytorch_pretrained_bert
- nltk
- flask

## Steps to Reproduce Results

### Step 1: build KG server

We use KG files processed by previous work *Yuncheng Hua, Yuan-Fang Li, Guilin Qi, Wei Wu, Jingyao Zhang, and Daiqing Qi. Less is more: Data-efficient complex question answering over knowledge bases. Journal of Web Semantics, 65:100612, 2020.* https://github.com/DevinJake/NS-CQA. 
We are unable to upload the KG files due to file size limitations, but you can download from the link above.

After downloading bfs_data.zip, please unzip it into data/bfs_data.

Then:
```
cd BFS
python save_reverse.py 
python server.py
```

### Step 2: preprocess data
```
cd data
python bert_emb.py
python mask.py
python retrieve.py
```
Note the full test set is particularly large, so we only upload a subset of it. You can get similar experimental results to those reported in the paper using this subset. The complete test dataset can be downloaded from https://github.com/DevinJake/NS-CQA and processed into a similar format

### Step 3: train rewriting model
``` 
python action2text.py # train a model that translates a action sequence into a query
python question_decompose.py --model_name=epoch_%d_score_%s.bin # build a rewrite dataset using trained model, please replace argument with name of the model having the highest score in saved_models/action2text/
python train_question_rewrite.py  # train a rewrite model
python predict_question_rewrite.py --model_name=epoch_%d_score_%s.bin # predict with a rewrite model
cd data
python bert_emb_ques.py # embedding rewrited question
```

### Step 4: train action sequence generation model
```
python train.py --name=pretrain_full --mode=pretrain --data_folder=data --log_folder=logs --model_folder=saved_models --symbol_file=data/symbol.txt --batch_size=32 --num_train_epochs=100  # pretrain
python train.py --name=rl_full --mode=rl --batch_size=8 --data_folder=data --log_folder=logs --model_folder=saved_models --symbol_file=data/symbol.txt --load_model=saved_models/pretrain/pretrain_full/model.bin --num_train_epochs=50 --learning_rate=1e-5 --seed=1234 --web_url=http://127.0.0.1:5577/post --reward_save_path=data/rl/reward_memory_adaptive.json # rl
```
It will take a long time to do reinforce learning.

### Step 5: predict
```
python predict_with_beam_search.py --name=predict_with_rl_full --data_folder=data/test/test_sample --load_model=saved_models/rl/rl_full/[model_name] --symbol_file=data/symbol.txt --sim_num=3 --web_url=http://127.0.0.1:5577/post --assistant_file=most_sim_10.json --reward_load_path=preprocess/test/test_sample/reward_memory_adaptive.json
```
please replace [model_name] with name of the model having the highest reward

### Step 6: evaluate
```
python calculate_sample_test_dataset.py predict_with_rl_full
```
Yout will find result file  data/test/test_sample/results/predict_with_rl_full_p.txt 
