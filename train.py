import os
import warnings
warnings.filterwarnings('ignore')
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

START = '[START]'
# SEP = '[SEP]'
EOS = '[EOS]'
PAD = '[PAD]'

from model import RNNEncoder, RNNDecoder, Seq2SeqModel
from train_util import load_data, train_with_target, train_RL
from utils import init_logger

if __name__ == "__main__":

    # command line parameters

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--mode", type=str, required=True, help="pretrain or rl")
    parser.add_argument("--data_folder",required=True, type=str, help="data folder for training. E.g., train")
    parser.add_argument("--log_folder", required=True, type=str, help="folder for storing logs")
    parser.add_argument("--model_folder", required=True, type=str, help="folder for storing trained models")
    parser.add_argument("--symbol_file", required=True, type=str, help="particular symbol of decoding")
    # model parameters
    parser.add_argument("--web_url", default='http://127.0.0.1:5577/post', type=str, help="kg exec url")
    parser.add_argument("--assistant_file", type=str, default=None, help="assitant json file")
    parser.add_argument("--emb_dim", type=int, default=768, help="The dimension of the word embeddings")
    parser.add_argument("--type_emb_dim", type=int, default=100, help="The dimension of the word embeddings")
    parser.add_argument("--hidden_dim", type=int, default=300, help="The dimension of the hidden states")
    parser.add_argument("--num_directions", default=2, type=int, help="whether to use a bidirectional RNN encoder")
    parser.add_argument("--num_layers", default=1, type=int, help="number of stacked layers")
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout value")
    parser.add_argument("--max_action_num", default=5, type=int, help="decoding limit")
    parser.add_argument("--max_action_len", default=7, type=int, help="decoding limit")
    parser.add_argument("--max_total_action_len", default=40, type=int, help="decoding limit")
    parser.add_argument("--sent_attn_norm", default='softmax', type=str, help="sent-level attention weights normalization method")
    parser.add_argument("--word_attn_norm", default='softmax', type=str, help="word-level attention weights normalization method")

    # Other parameters
    parser.add_argument("--do_train", default=True, type=bool, help="whether to do train")
    parser.add_argument("--do_eval", default=True, type=bool, help="whether to do eval")
    parser.add_argument("--bert_finetune", default=False, type=bool, help="whether to finetune bert")
    parser.add_argument("--tf_folder", default=None, type=str, help="folder for storing tensorboard logs")
    parser.add_argument("--load_model", default=None, type=str, help="The pre-trained model to load")
    parser.add_argument("--reward_save_path", default=None, type=str, help="action reward to save")
    parser.add_argument("--adaptive", default=True, type=bool, help="whether to use adaptive reward")
    # parser.add_argument("--reward_load_path", default=None, type=str, help="action reward to load")
    # parser.add_argument("--memory_buffer_json", default=None, help="Load the recorded action memory for CHER training")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="The epoches of training")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learing rate of training")
    parser.add_argument("--gamma", default=0.9, type=float, help="reward decay rate")
    parser.add_argument("--seed", default=1234, type=int, help="random seeed for initialization")
    parser.add_argument("--reward_lambda", default=0.0, type=float, help="weight of biaes reward")

    args = parser.parse_args()

    #  get path 
    data_path = os.path.join(args.data_folder, args.mode)

    log_path = os.path.join(args.log_folder, args.mode, args.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    tf_path = None
    if args.tf_folder is not None:
        tf_path = os.path.join(args.tf_folder, args.mode, args.name)
        if not os.path.exists(tf_path):
            os.makedirs(tf_path, exist_ok=True)

    save_model_path = os.path.join(args.model_folder, args.mode, args.name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path, exist_ok=True)
    save_model_file = os.path.join(save_model_path, 'model.bin')

    logger = init_logger(os.path.join(log_path, 'log.txt'))
    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(device)
    else:
        device = torch.device('cpu')
        logger.error("cuda is unavailable")
    # device = torch.device('cpu')device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load symbol
    symbol_lst = []
    for line in open(args.symbol_file).readlines():
        symbol, type_id = line.strip().split()
        symbol_lst.append(symbol)

    # build model
    encoder = RNNEncoder(args)
    decoder = RNNDecoder(args, device, len(symbol_lst), 5,  symbol_lst.index(START), \
            symbol_lst.index(EOS), symbol_lst.index(PAD))
    model = Seq2SeqModel(device, encoder, decoder)
    # load pretrained model
    load_model_file = args.load_model if args.load_model else None
    if load_model_file != None and os.path.exists(load_model_file):
        model_dic = torch.load(load_model_file, map_location='cpu')
        model.load_state_dict(model_dic)
        logger.info("successfully load pre-trained model ...")
    else:
        logger.info("pre-trained model not found...")

    model.to(device)
    

    # load train data
    data = load_data(data_path, args, load_emb=not args.bert_finetune)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    tf_writer = None
    if tf_path is not None:
        tf_writer = SummaryWriter(log_dir=tf_path)
        logger.info('tf writer path: %s' % tf_path)

    id2name = {}
    # train
    if args.mode == 'pretrain':
        train_with_target(device, model, optimizer, data, symbol_lst, symbol_lst.index(START), 
                            symbol_lst.index(EOS), symbol_lst.index(PAD), logger, tf_writer, save_model_file,  
                            epoch_num=args.num_train_epochs, batch_size=args.batch_size, do_train=args.do_train, 
                            do_eval=args.do_eval, finetune_bert=args.bert_finetune, id2name=id2name)
    else:
        reward_memory = {}
        if args.reward_save_path is not None and os.path.exists(args.reward_save_path):
            reward_memory = json.load(open(args.reward_save_path))
            logger.info("Loading reward buffer with size %d from %s..." % (len(reward_memory), args.reward_save_path))

        memomy_buffer = {}
        memory_buffer_json = os.path.join(save_model_path, 'action_memory_buffer.json')
        if os.path.exists(memory_buffer_json):
            logger.info("Loading the stored action memory from %s..." % os.path.join(save_model_path, 'action_memory_buffer.json'))
            memory_buffer = json.load(open(os.path.join(save_model_path, 'action_memory_buffer.json')))
        
        train_RL(device, model, optimizer, data, symbol_lst, symbol_lst.index(START), symbol_lst.index(EOS), symbol_lst.index(PAD), \
                logger, tf_writer, save_model_path, args.reward_save_path, memory_buffer_json, reward_memory, memomy_buffer, args.adaptive, \
                args.web_url, epoch_num=args.num_train_epochs, batch_size=args.batch_size, do_train=args.do_train, do_eval=args.do_eval, \
                reward_lambda=args.reward_lambda)
    

    