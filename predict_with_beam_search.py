import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import json

START = '[START]'
# SEP = '[SEP]'
EOS = '[EOS]'
PAD = '[PAD]'

from model import RNNEncoder, RNNDecoder, Seq2SeqModel
from train_util import load_data, predict
from utils import init_logger


if __name__ == "__main__":

    # command line parameters

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_folder",required=True, type=str, help="test data folder")
    parser.add_argument("--name", required=True, type=str, help="file to strore predict results")
    parser.add_argument("--load_model", required=True, type=str, help="The trained model to load")
    parser.add_argument("--symbol_file", required=True, type=str, help="particular symbol of decoding")
    parser.add_argument("--web_url", required=True, type=str, help="kg exec url")
    parser.add_argument("--sim_num", required=True, type=int, help="number of similar qa pair")

    # model parameters
    parser.add_argument("--top_k", type=int, default=5, help="candidates number")
    parser.add_argument("--assistant_file", type=str, default=None, help="assitant json file")
    parser.add_argument("--reward_load_path", default=None, type=str, help="action reward to save")
    parser.add_argument("--emb_dim", type=int, default=768, help="The dimension of the word embeddings")
    parser.add_argument("--type_emb_dim", type=int, default=100, help="The dimension of the word embeddings")
    parser.add_argument("--hidden_dim", type=int, default=300, help="The dimension of the hidden states")
    parser.add_argument("--num_directions", default=2, type=int, help="whether to use a bidirectional RNN encoder")
    parser.add_argument("--num_layers", default=1, type=int, help="number of stacked layers")
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout value")
    parser.add_argument("--max_action_num", default=5, type=int, help="decoding limit")
    parser.add_argument("--max_action_len", default=7, type=int, help="decoding limit")
    parser.add_argument("--max_total_action_len", default=30, type=int, help="decoding limit")
    parser.add_argument("--sent_attn_norm", default='softmax', type=str, help="sent-level attention weights normalization method")
    parser.add_argument("--word_attn_norm", default='softmax', type=str, help="word-level attention weights normalization method")

    # Other parameters
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size")
    parser.add_argument("--seed", default=1234, type=int, help="random seeed for initialization")
    parser.add_argument("--bert_finetune", default=False, type=bool, help="whether to finetune bert")

    args = parser.parse_args()

    logger = init_logger()

    logger.info(args)

    #  get path 
    data_path = args.data_folder


    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Use {} gpus!".format(torch.cuda.device_count()))
    else:
        device = torch.device('cpu')
        logger.error("cuda is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load symbol
    symbol_lst = []
    type_lst = []
    for line in open(args.symbol_file).readlines():
        symbol, type_id = line.strip().split()
        symbol_lst.append(symbol)

    # build model
    encoder = RNNEncoder(args)
    decoder = RNNDecoder(args, device, len(symbol_lst), 5,  symbol_lst.index(START), \
            symbol_lst.index(EOS), symbol_lst.index(PAD))
    model = Seq2SeqModel(device, encoder, decoder)
    # load pretrained model
    load_model_file = args.load_model
    model_dic = torch.load(load_model_file, map_location='cpu')
    model.load_state_dict(model_dic)
    logger.info("successfully load pre-trained model ...")
    model.to(device)
    

    # load test data
    data = load_data(data_path, args, load_emb=not args.bert_finetune)
    # data = (data[0][576:], data[1][576:], data[2][576:])

    output_path = open(os.path.join(data_path, 'results', args.name + '.txt'), 'w')
    logger.info('write results to:' + os.path.join(data_path, 'results', args.name + '.txt'))

    id2name = {}
    # predict
    reward_memory = {}
    if args.reward_load_path is not None and os.path.exists(args.reward_load_path):
        reward_memory = json.load(open(args.reward_load_path))
        logger.info("Loading reward buffer with size %d from %s..." % (len(reward_memory), args.reward_load_path))
    predict(device, model, data, symbol_lst, symbol_lst.index(START), symbol_lst.index(EOS), 
        symbol_lst.index(PAD), logger, output_path, args.batch_size, args.bert_finetune, id2name, \
            True, args.top_k, reward_memory, args.web_url)
    json.dump(reward_memory, open(args.reward_load_path, 'w'), ensure_ascii=False)


    