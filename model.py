import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import sparsemax

epsilon = 1e-30


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


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

def get_batch_results_from_bert(device, tokens_list, tokenizer, bert):
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
    input_tensor = input_tensor.to(device)
    attention_tensor = attention_tensor.to(device)
    with torch.no_grad():
        bert_output = bert(input_ids=input_tensor, attention_mask=attention_tensor, return_dict=False) 
    return bert_output

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.

    """

    def __init__(self, dim, method='softmax'):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.linear_in = nn.Linear(dim, dim, bias=False)       
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        assert method in ['softmax', 'sparsemax']
        if method  == 'softmax':
            self.normalize = nn.Softmax(dim=-1)
        else:
            self.normalize = sparsemax


    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        assert src_dim==tgt_dim, "Dimension do not match! (%d vs %d)" % (src_dim, tgt_dim)

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)

    def forward(self, source, memory_bank, memory_lengths=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[batch x dim]`
          * Attention distribtutions for each query
             `[batch x src_len]`
        """

        # [1 x batch x dim] -> [batch x 1 x dim]
        source = source.transpose(0, 1)

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size() 

        # compute attention scores, as in Luong et al.
        # [batch x 1 x src_len]
        align = self.score(source, memory_bank)  # get attention werights by bilinear similarity

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(~mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        align_vectors = self.normalize(align.view(batch*target_l, source_l))
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        # [batch, target_l, dim]
        c = torch.bmm(align_vectors, memory_bank)
        
        # # concatenate
        concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        # [batch, dim]
        attn_h = torch.tanh(attn_h).squeeze(1)

        # [batch, src_len]
        align_vectors = align_vectors.squeeze(1)

        return attn_h, align_vectors

class RNNEncoder(nn.Module):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, opt):
        super(RNNEncoder, self).__init__()

        hidden_size = opt.hidden_dim // opt.num_directions

        self.rnn  = nn.LSTM(input_size=opt.emb_dim,
                        hidden_size=hidden_size,
                        num_layers=opt.num_layers,
                        dropout=opt.dropout,
                        batch_first=True,
                        bidirectional=opt.num_directions==2)

        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim
        # Initialize the bridge layer

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()"

        # s_len, batch, emb_dim = src.size()

        packed_emb = src
        if lengths is not None:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack_padded_sequence(src, lengths_list, batch_first=True)
        
        # hidden_size here means the inputting param of RNN
        # [source_len, batch, hidden_size* direction], ([num_layers * direction, batch, hidden_size], [num_layers * direction, batch, hidden_size])
        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None:
            memory_bank = pad_packed_sequence(memory_bank)[0]

        return encoder_final, memory_bank, lengths

class RNNDecoder(nn.Module):
    """
    Hierarchical recurrent attention-based decoder class.
    """

    def __init__(self, opt, device, symbol_num=20, type_num=4, start_idx=1, eos_idx=3, pad_idx=0, dropout=0.0):
        super(RNNDecoder, self).__init__()

        # Basic attributes.
        self.device = device
        self.start_idx = start_idx # start token
        # self.sep_idx = sep_idx # end of word-level decoding
        self.eos_idx = eos_idx # end of sent=level decoding
        self.pad_idx = pad_idx
        self.symbol_num = symbol_num # (A1-A16, START, SEP, EOS, PAD, &, |, -)
        self.embeddings = nn.Embedding(symbol_num, opt.emb_dim) # embdeddings of 16 functions and 4 particular symbols, 768
        self.type_embeddings = nn.Embedding(type_num, opt.type_emb_dim) # 100
        self.bidirectional_encoder = opt.num_directions == 2
        self.num_layers = opt.num_layers
        self.hidden_size = opt.hidden_dim # 300
        self.type_emb_size = opt.type_emb_dim
        self.max_action_num = opt.max_action_num
        self.max_action_len = opt.max_action_len
        self.max_total_action_len = opt.max_total_action_len

        # Decoder state
        self.sent_state = {}
        self.word_state = {}

        # Build the sentence decoder RNN.
        # h_i = LSTM(h~_i-1_end, h_i-1)
        self.dropout = nn.Dropout(dropout)

        # Build the word decoder RNN
        # h_i_j = LSTM([h~_i_j-1; e_last], h_i_j-1)
        self.dec_rnn = nn.LSTM(input_size=opt.hidden_dim+opt.emb_dim+opt.type_emb_dim,               
                                    hidden_size=opt.hidden_dim,
                                    num_layers=opt.num_layers,
                                    dropout=opt.dropout)
        
        # output logits of particular symbols
        self.linear_out = nn.Linear(opt.hidden_dim, symbol_num) 
        # whether to ouput candidate vocabulary
        self.linear_copy = nn.Linear(opt.hidden_dim, 1)
        # feature mapping
        self.linear = nn.Linear(opt.hidden_dim, opt.emb_dim + opt.type_emb_dim)

        self.word_attn = GlobalAttention(opt.hidden_dim, opt.word_attn_norm)
        


    def init_state(self, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim. [2 x b x 150]
            # We need to convert it to layers x batch x (directions*dim). [1 x b x 300]
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)

            # last encoder state initialization
            hidden = hidden[-1].unsqueeze(0)

            return hidden
        
        self.state = {}
        # (h, c)
        batch_size = encoder_final[0].size(1)
        h_size = (batch_size, self.hidden_size)

        # Init the sentence hidden state with encoder final hidden. (h, c)
        self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final])

        # Init the sentence input feed with zero. [batch_size x self.hidden_size]
        self.state["input_feed"] = \
            self.state["hidden"][0][0].data.new(*h_size).zero_()


    def forward(self, tgt, tgt_emb, memory_banks, memory_lengths=None, coverage=None):
    
        # Run the forward pass of the RNN.
        tgt_batch, tgt_s_len = tgt.size()
        # [tgt_s_len x batch]
        tgt = tgt.transpose(0, 1).contiguous()
        # [tgt_s_len x batch x embedding_size]
        emb = tgt_emb.transpose(0, 1).contiguous()

        # Initialize local and return variables.
        dec_outs = []
        attns = []
        coverages = [coverage]

        # sent level decoding
        for i, word_emb_i in enumerate(emb.split(1, dim=0)):  
            # [tgt_batch x embedding_size]
            word_emb_i = word_emb_i.squeeze(0)
            word_input_feed = self.state["input_feed"]
            dec_state = self.state["hidden"]  # last hidden state 
            decoder_input = torch.cat([word_emb_i, word_input_feed], 1).unsqueeze(0) # [seq_len=1,batch,feature_len]
            # rnn_output: [1 x batch x self.hidden]
            # dec_state: LSTM (h, c) 
            rnn_output, dec_state = self.dec_rnn(decoder_input, dec_state)
            # s_t_hat = torch.cat((dec_state[0].view(-1, self.hidden_size),
            #                  dec_state[1].view(-1, self.hidden_size)), 1)  # batch x 2*hidden_dim
            
            # sent_attn_h: [batch x self.hidden], 
            # sent_p_attn: [batch x seq_len]
            # attn_h, p_attn = self.word_attn(rnn_output,
            #                                 memory_bank=memory_banks,
            #                                 memory_lengths=memory_lengths)
            attn_h, p_attn = self.word_attn(rnn_output,
                                            memory_bank=memory_banks,
                                            memory_lengths=memory_lengths)
            # update the sent_states here for sent_state
            self.state["hidden"] = dec_state
            self.state["input_feed"] = attn_h
            dec_outs += [attn_h]
            attns += [p_attn]

        # [tgt_s_len x batch x h_size]
        dec_outs = torch.stack(dec_outs)
        dec_outs = dec_outs.view(tgt_s_len, tgt_batch, -1)

        # [tgt_s_len x batch x seq_len]
        attns = torch.stack(attns)
        attns = attns.view(tgt_s_len, tgt_batch, -1)
        
        return dec_outs, attns

    def decode_one(self, dec_input, dec_state, vocab_emb, vocab_size, memory_banks, memory_lengths=None): 
        '''
        decoder with one step
        Args:
            dec_input (`Tensor`): input_embedding + attentional src context
                `[batch_size*beam_width x feature_dim]` 
            dec_state (`Tuple`) 
            vocab_emb (`Tensor`): embeddings of candidate vocab tokens `
                padding with [batch x n_token x embedding]` 
            vocab_size(`List`): valid vocab lengths with size `[batch]`
        Returns:
            probs (`FloatTensor`): probabilities of output tokens
                `[batch_size*beam_width x vocab_size]` 
            dec_state (`Tuple`) 
            attn_h (`FloatTensor`):
        '''
        batch_size = vocab_emb.size(0)
        rnn_output, dec_state = self.dec_rnn(dec_input, dec_state)

        # [batch_size*beam_width x self.hidden]
        attn_h, _ = self.word_attn(rnn_output,
                                        memory_bank=memory_banks,
                                        memory_lengths=memory_lengths)          
                 
        logits_s = self.linear_out(attn_h)
        logits_s[:, self.pad_idx] = -float('inf')
        probs_s = torch.softmax(logits_s, dim=1)

        p_copy = torch.sigmoid(self.linear_copy(attn_h))

        # [batch_size x beam_size x emb_dim] x [batch_size x emb_dim x n_token]
        # ->  [batch_size x beam_size x n_token]
        attn_h = attn_h.reshape(batch_size, -1, attn_h.size(-1))  
        beam_width = attn_h.size(1)     
        logits = torch.bmm(self.linear(attn_h), vocab_emb.transpose(1,2)).reshape(batch_size*beam_width, -1)
        for idx in range(vocab_size.size(0)):
            logits[idx*beam_width:(idx+1)*beam_width, vocab_size[idx]:] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        
        probs = torch.cat( [torch.mul((1-p_copy), probs_s), torch.mul(p_copy, probs)], dim=1) +  epsilon    

        
        return probs, dec_state, attn_h


    def beam_decode(self, vocab_emb, vocab_size, encoder_final, memory_banks, memory_lengths=None, beam_size=10, top_k=5):
        """
        beam search！
        Args:
            memory_banks (`Tensor`): the memory banks from encoder
                `[batch x src_len x hidden]` 
            memory_lengths (`LongTensor`): the source lengths with size `[batch]`
            vocab_emb(List of 'Tensor'): embeddings of candidate decoding tokens `
                                padding with [batch x n_token x embedding]` 
            vocab_size(`List`): valid vocab lengths with size `[batch]`
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * actions: list of decoding action sequence 
                         `[batch x top_k x tgt_s_len]`.
                * logits: predict log_probs
                        `[ batch x top_k x tgt_s_num x tgt_s_len]`.
        """

        batch_size = memory_banks.size(0)
        max_vocab_size = self.symbol_num+vocab_emb.size(1)

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)] 
        results["scores"] = [[] for _ in range(batch_size)]  
        results["probs"] = [[] for _ in range(batch_size)]  
    
        # repeat for beam_size
        memory_banks = tile(memory_banks, beam_size, dim=0)
        memory_lengths = tile(memory_lengths, beam_size, dim=0)

        self.init_state(encoder_final)
        dec_state = (tile(self.state["hidden"][0], beam_size, dim=1), tile(self.state["hidden"][1], beam_size, dim=1)) 
        attn_context = tile(self.state["input_feed"], beam_size, dim=0).unsqueeze(0)
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=self.device)
        beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=self.device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_idx,
            dtype=torch.long,
            device=self.device)

        # [batch x beam_size]
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),  
                         device=self.device).repeat(batch_size, 1)) 
        
        # [batch*beam_size x 1]
        topk_prob_list = (
            torch.tensor([1.0] + [0.0] * (beam_size - 1),  
                         device=self.device).repeat(batch_size)).unsqueeze(1)
        hypotheses = [[] for _ in range(batch_size)]
        stored_topk_score = [ [] for _ in range(batch_size)]
        for word_step in range(self.max_total_action_len - 1): # contain [start]
            word_decoder_input = alive_seq[:, -1].view(-1, 1)
            word_decoder_input_fixed = torch.where(word_decoder_input < self.symbol_num, word_decoder_input, 0)
            word_decoder_input_embedding = self.embeddings(word_decoder_input_fixed)
            word_decoder_input_embedding = torch.cat([word_decoder_input_embedding, self.type_embeddings(torch.zeros_like(word_decoder_input))], dim=-1)
            for i in range(word_decoder_input.size(0)):
                if word_decoder_input[i, 0].item() >= self.symbol_num:
                    word_decoder_input_embedding[i, 0] = vocab_emb[i//beam_size, word_decoder_input[i, 0]-self.symbol_num]
            word_decoder_input_embedding = word_decoder_input_embedding.transpose(0, 1).contiguous()
            # [1, batch_size*beam_size, emb_dim+hidden_dim]
            dec_input = torch.cat([word_decoder_input_embedding, attn_context], dim=2)
            # `[batch_size*beam_width x max_vocab_size]` 
            probs, dec_state, attn_h = self.decode_one(dec_input, dec_state, vocab_emb, vocab_size, memory_banks, memory_lengths)
            attn_context = attn_h.reshape(1, -1, attn_h.size(-1))

            # Multiply probs by the beam probability.  [batch_size*beam_size, max_vocab_size]
            log_probs = probs.log() + topk_log_probs.view(-1).unsqueeze(1)
            curr_scores = log_probs.reshape(-1, beam_size * max_vocab_size) # [batch, beam*max_vocab_size]
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1) # [batch, beam]
            topk_log_probs = topk_scores
            selected_probs = probs.reshape(-1, beam_size * max_vocab_size).gather(-1, topk_ids)
            # Resolve beam origin and true word ids.
            topk_beam_index = torch.div(topk_ids, max_vocab_size, rounding_mode='floor') # [batch, beam_size] belongs to  which prev beam
            topk_ids = topk_ids.fmod(max_vocab_size)   # [batch, beam_size]   top_k ids of vocab
            
            # Map beam_index to batch_index in the flat representation.  + batch_index * beam_size
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)  # [batch*beam_size]

            # Append last prediction. # [batch*beam_size, seq_len+1]
            temp_alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            temp_topk_prob_list = torch.cat(
                [topk_prob_list.index_select(0, select_indices),
                 selected_probs.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.eos_idx)
            if word_step + 1 == self.max_total_action_len - 1:
                is_finished.fill_(1)
            # end_condition = is_finished[:, 0].eq(1) # [batch_size]
            end_condition = torch.full([is_finished.size(0)], False, device=self.device)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = temp_alive_seq.view(-1, beam_size, temp_alive_seq.size(-1))
                pred_probs = temp_topk_prob_list.view(-1, beam_size, temp_topk_prob_list.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]  # true batch index
                    # if end_condition[i]:
                    #     is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j], # sum(logp)
                            predictions[i, j], 
                            pred_probs[i, j])) # prob listk, 
                        if len(stored_topk_score[b]) < top_k:
                            stored_topk_score[b].append(topk_scores[i, j].item())
                        else:  # update topk score
                            min_index = np.argmin(stored_topk_score[b])
                            if topk_scores[i, j].item() > stored_topk_score[b][min_index]:
                                stored_topk_score[b][min_index] = topk_scores[i, j].item()
                    
                    # If the batch reached the end, save the n_best hypotheses.
                    if (len(stored_topk_score[b]) >= top_k and min(stored_topk_score[b]) >= topk_scores[i, 0]) or \
                        word_step + 1 ==  self.max_total_action_len - 1:
                        end_condition[i] = True
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for k in range(min(top_k, len(best_hyp))):
                            results["scores"][b].append(best_hyp[k][0])
                            results["predictions"][b].append(best_hyp[k][1])
                            results["probs"][b].append(best_hyp[k][2])
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break

                # we need to remove suquence with eos symbol, sort again
                log_probs[:, self.eos_idx] = float("-inf")
                curr_scores = log_probs.reshape(-1, beam_size * max_vocab_size) # [batch, beam*max_vocab_size]
                topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1) # [batch, beam]
                selected_probs = probs.reshape(-1, beam_size * max_vocab_size).gather(-1, topk_ids)
                topk_beam_index = torch.div(topk_ids, max_vocab_size, rounding_mode='floor') # [batch, beam_size] belongs to  which prev beam
                topk_ids = topk_ids.fmod(max_vocab_size)   # [batch, beam_size]   top_k ids of vocab
                batch_index = (
                        topk_beam_index
                        + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
                select_indices = batch_index.view(-1)  # [batch*beam_size]
                alive_seq = torch.cat(
                    [alive_seq.index_select(0, select_indices),
                    topk_ids.view(-1, 1)], -1)
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                topk_prob_list = torch.cat(
                    [topk_prob_list.index_select(0, select_indices),
                    selected_probs.view(-1, 1)], -1)
                pred_probs = topk_prob_list.view(-1, beam_size, topk_prob_list.size(-1))

                # Remove finished batches for the next step.
                topk_log_probs = topk_scores.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)  # [non_finish_batch, beam_size]
                batch_offset = batch_offset.index_select(0, non_finished)
                vocab_emb = vocab_emb.index_select(0, non_finished)
                vocab_size = vocab_size.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
                topk_prob_list = pred_probs.index_select(0, non_finished) \
                    .view(-1, topk_prob_list.size(-1))
            
            else:
                alive_seq = temp_alive_seq
                topk_prob_list = temp_topk_prob_list
            # Reorder states.
            select_indices = batch_index.view(-1)
            memory_banks = memory_banks.index_select(0, select_indices)
            memory_lengths = memory_lengths.index_select(0, select_indices)
            attn_context = attn_context.index_select(1, select_indices)
            dec_state = (dec_state[0].index_select(1, select_indices), dec_state[1].index_select(1, select_indices)) 
        
        return results


    def forward_with_no_teacher(self, vocab_emb, vocab_size, memory_banks, memory_lengths=None):
        batch_size, seq_len, _ = memory_banks.size()

        results = {}
        results["predictions"] = []
        results["scores"] = []
        results["attention"] = []


        word_seq_so_far = torch.full(
            [batch_size, 1], self.start_idx, dtype=torch.long).to(self.device)
        word_alive_attn = None
        word_dec_finished = torch.zeros(batch_size, dtype=torch.uint8).to(self.device)
        # init with zero tensor
        coverage = torch.zeros((memory_banks.size(0), memory_banks.size(1))).to(self.device) 
            
        for word_step in range(self.max_total_action_len - 1): # contain [start]
            # last decoder output, [batch x 1]
            word_decoder_input = word_seq_so_far[:, -1].view(-1, 1)

            # check whether current word decoding is finished
            word_dec_finished = word_dec_finished | (word_decoder_input == self.eos_idx).view(-1)

            if  word_dec_finished.sum().item() == batch_size:
                break   # last one is sep
            
            # get decoder input  embeddings   [batch x 1 x emb_size]
            word_decoder_input_fixed = torch.where(word_decoder_input < self.symbol_num, word_decoder_input, 0)
            word_decoder_input_embedding = self.embeddings(word_decoder_input_fixed)
            word_decoder_input_embedding = torch.cat([word_decoder_input_embedding, self.type_embeddings(torch.zeros_like(word_decoder_input))], dim=-1)
            for i in range(word_decoder_input.size(0)):
                if word_decoder_input[i, 0].item() >= self.symbol_num:
                    word_decoder_input_embedding[i, 0] = vocab_emb[i, word_decoder_input[i, 0]-self.symbol_num]
        
            # one step decoding
            # dec_out: [1 x  batch x h_size]
            # dec_attn: [1 x  batch x seq_len]
            # next_coverage: [2 x  batch x seq_len]
            dec_out, dec_attn = self.forward(word_decoder_input, 
                                            word_decoder_input_embedding,
                                            memory_banks,
                                            memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)
            
            # probs of output function and other symbols                                
            # [batch x n_symbol]
            logits_s = self.linear_out(dec_out)
            logits_s[:, self.pad_idx] = -float('inf')
            probs_s = torch.softmax(logits_s, dim=1)
            # probs of output entitiy, relation, type and num [batch x 1 x h_size] x [batch x h_size x max_vocab_num]
            logits = torch.bmm(self.linear(dec_out).unsqueeze(1), vocab_emb.transpose(1,2)).squeeze(1)
            # set unvalid token prob
            for idx in range(batch_size):
                logits[idx, vocab_size[idx]:] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            # prob to choose output entitiy, relation, type and num [batch x 1]
            p_copy =  torch.sigmoid(self.linear_copy(dec_out))
            probs = torch.cat( [torch.mul((1-p_copy), probs_s), torch.mul(p_copy, probs)], dim=1)       
            # probs = probs_s
            topk_scores, topk_ids = probs.topk(1, dim=-1)
            word_seq_so_far = torch.cat([word_seq_so_far, topk_ids.view(-1, 1)], -1)
            # Append attention
            current_attn = dec_attn.view(1, batch_size, -1)
            if word_alive_attn is None:
                # [1, 1, batch, s_len]
                word_alive_attn = current_attn
            else:
                # [1, word_step+1, batch, s_len]
                word_alive_attn = torch.cat([word_alive_attn, current_attn], 0)
        
        # word_seq_so_far = word_seq_so_far[:, :, 1:]
        orig_action_len = word_seq_so_far.size(-1)

        # pad word_seq_so_far with max_action_len
        pad_tensors = torch.full([batch_size, self.max_total_action_len - orig_action_len], self.pad_idx,
                                    dtype=word_seq_so_far.dtype).to(self.device)
        # [batch_size, 1, max_action_len]
        word_seq_so_far = torch.cat([word_seq_so_far, pad_tensors], dim=-1)

        # pad word_alive_attn with max_action_len
        pad_tensors = torch.full([self.max_total_action_len - orig_action_len, batch_size, word_alive_attn.size(-1)],
                                    0.0, dtype=word_alive_attn.dtype).to(self.device)
        # [1, max_action_len, batch, s_len]
        word_alive_attn = torch.cat([word_alive_attn, pad_tensors], dim=0)

        for i in range(topk_scores.size(0)):
            # Store finished hypotheses for this batch. Unlike in beam search,
            # there will only ever be 1 hypothesis per example.
            # [1]
            score = topk_scores[i, 0]
            # [batch_size, max_action_len]
            pred = word_seq_so_far[i]
            # [max_action_len, batch, s_len]
            attn = word_alive_attn[:, i, :]

            results["scores"].append(score)
            results["predictions"].append(pred)
            results["attention"].append(attn)
        
        return results

    def forward_with_teacher(self, tgt, tgt_emb, vocab_emb, vocab_size, memory_banks, memory_lengths=None):
        # remove end symbol
        tgt = tgt[:, :-1]
        if tgt_emb is not None:
            tgt_emb = tgt_emb[:, :-1, :]
        dec_outs, attns = self.forward(tgt, tgt_emb, memory_banks, memory_lengths=memory_lengths)
        # [(tgt_s_num*tgt_s_len) x batch x h_size] -> [batch x (tgt_s_num*tgt_s_len) x h_size]
        _, batch_size, _ = dec_outs.size()
        dec_outs = dec_outs.transpose(0, 1).contiguous()
        # probs of output function and other symbols [n x n_symbol]                               
        logits_s = self.linear_out(dec_outs)
        # [batch x (tgt_s_num*tgt_s_len) x n_symbol]-> [n x n_symbol]
        logits_s = logits_s.view(-1, logits_s.size(2))
        logits_s[:, self.pad_idx] = -float('inf')
        probs_s = torch.softmax(logits_s, dim=1)
        # probs of output entitiy, relation, type and num 
        # [batch x (tgt_s_num*tgt_s_len) x emb_size] x [batch x emb_size x n_token]
        # -> [batch x (tgt_s_num*tgt_s_len) x n_token]
        logits = torch.bmm(self.linear(dec_outs), vocab_emb.transpose(1,2))
        # set unvalid token prob
        for idx in range(batch_size):
            logits[idx, :, vocab_size[idx]:] = -float('inf')
        logits = logits.view(-1, logits.size(2))
        probs = torch.softmax(logits, dim=-1)
        # prob to choose output entitiy, relation, type and num [batch x 1]
        p_copy =  torch.sigmoid(self.linear_copy(dec_outs.view(-1, dec_outs.size(2))))
        probs = torch.cat( [torch.mul((1-p_copy), probs_s), torch.mul(p_copy, probs)], dim=1)
        # probs = probs_s
        # [batch x tgt_s_len x seq_len]
        attns = attns.transpose(0, 1).contiguous()
        return probs, attns


class Seq2SeqModel(nn.Module):
    """
    Args:
      encoder (:obj:`EncoderBase`): a flattened or hierarchical encoder object
      decoder (:obj:`RNNDecoderBase`): a flattened or hierarchical decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, device, encoder, decoder, bert_model=None, tokenizer=None):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
    
    # encode question and vocabulary
    def bert_encode(self, question, vocab_list):
        tokens_list = []
        tokens_list.append(['[CLS]'] + self.tokenizer.tokenize(question) + ['[SEP]'])
        ques_len = len(tokens_list[0])
        target_idx_list = []
        for vocab_name in vocab_list:
            tokens, target_idx = prepare_sentence_for_bert(question, vocab_name, self.tokenizer)
            tokens_list.append(tokens)
            target_idx_list.append(target_idx)
        # ([batch_size, sequence_length, hidden_size], [batch_size, hidden_size])
        batch_model_results = get_batch_results_from_bert(self.device, tokens_list, self.tokenizer, self.bert_model)
        # sent emb
        question_emb = batch_model_results[0][0][1:ques_len-1]
        # vocab_emb 
        vocab_emb = None
        for i, (start, end) in enumerate(target_idx_list):
            if vocab_emb is None:
                vocab_emb =  batch_model_results[0][i+1, start:end, :].mean(dim=0).unsqueeze(0)
            else:
                vocab_emb = torch.cat([vocab_emb, batch_model_results[0][i+1, start:end, :].mean(dim=0).unsqueeze(0)], dim=0)
        return question_emb, vocab_emb


    def forward(self, src, lengths=None, tgt=None, tgt_emb=None, vocab_emb=None, vocab_type=None, vocab_size=None,testing=False, \
            beam_size=10, top_k=5, beam_search=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder of size `[src_length x batch x features]`.
            tgt (:obj:`LongTensor`):
                a target sequence of size `[batch x tgt_s_num x tgt_s_len]` for hr decoder.
            tgt_emb (`Tensor`): 
                a target sequence embeddings of size `[batch x tgt_s_num x tgt_s_len x nfeats]`.
            lengths(:obj:`LongTensor'):
                length of source sequences.
            vocab_emb(:obj:`Tensor`):
                embeddings of candidate entity, relation and types. [batch x max_vocab_num x nfeats]
            vocab_type(:obj:`list`):
                types of vocab_size. [[]]
            vocab_size(:obj:`LongTensor`):
                number of actual vocabs, [batch]

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        # encode src
        enc_state, memory_banks, lengths = self.encoder(src, lengths)
        # -> [batch x src_len x hidden_dim]
        memory_banks = memory_banks.transpose(0, 1).contiguous()
        # initialize decoder state
        self.decoder.init_state(enc_state)
        if vocab_type is not None:
            type_emb = torch.zeros((vocab_emb.size(0), vocab_emb.size(1), self.decoder.type_emb_size), dtype=vocab_emb.dtype).to(self.device) # [batch, vocab_size, type_emb_size]
            for i in range(len(vocab_type)):
                type_emb[i, :vocab_size[i]] = self.decoder.type_embeddings(torch.tensor(vocab_type[i]).to(self.device))
        vocab_emb = torch.cat([vocab_emb, type_emb], dim=-1)
        # decode
        if testing:
            if beam_search:
                return self.decoder.beam_decode(vocab_emb, vocab_size, enc_state, memory_banks, lengths, beam_size, top_k)
            results = self.decoder.forward_with_no_teacher(vocab_emb, vocab_size, memory_banks, lengths)
            return results
        else:
            # RL training
            if tgt == None:
                argmax_results = self.decoder.forward_with_no_teacher(vocab_emb, vocab_size, memory_banks, lengths)
                beam_results =  self.decoder.beam_decode(vocab_emb, vocab_size, enc_state, memory_banks, lengths, beam_size, top_k)
                return argmax_results, beam_results
            # supervised training
            else:
               return self.decoder.forward_with_teacher(tgt, tgt_emb,  vocab_emb, vocab_size,  memory_banks, memory_lengths=lengths)

            
