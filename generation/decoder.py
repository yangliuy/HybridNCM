import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def repeat(input, dim, k):
    """Repeat the input tensor k times along the dim dimention
        input: [dim, d]
        output: [dim*k, d]
    """
    if type(input) is tuple:
        size = [-1] * len(input[0].size())
        size.insert(dim, k)
        new_size = list(input[0].size())
        new_size[dim-1] *= k
        input = list(input)
        for idx in range(len(input)):
            input[idx] = input[idx].unsqueeze(dim).expand(tuple(size)).contiguous().view(new_size)
        input = tuple(input)
    else:
        size = [-1] * len(input.size())
        size.insert(dim, k)
        new_size = list(input.size())
        new_size[dim-1] *= k
        input = input.unsqueeze(dim).expand(tuple(size)).contiguous().view(new_size)
    return input

def resize(input, size):
    if type(input) is tuple:
        input = list(input)
        for idx in range(len(input)):
            input[idx] = input[idx].view(size)
        input = tuple(input)
    else:
        input.view(size)
    return input

def topK_2d(score):
    """
    Args: 
        score: [batch, beam_size, num_vocab]
    Return:
        top_score: [batch, beam_size], select score
        top_rowid: [batch, beam_size], beam id
        top_colid: [batch, beam_size], word id
    """
    batch_size, beam_size, num_vocab = score.size()
    flat_score = score.view(batch_size, beam_size * num_vocab)
    top_score, top_index = flat_score.topk(k=beam_size, dim=1)
    top_rowid, top_colid = top_index / num_vocab, top_index % num_vocab 
    return top_score, top_rowid, top_colid  

def select_hid(hidden, batch_id, row_id):
    """ Re-arange the hidden state according to the selected beams in the previous step
    Args:
        hidden: [batch*beam_size, hidden_size] 
        batch_id: [batch, beam_size] 
        row_id: [batch, beam_size] 
    Return:
        new_hidden: [batch*beam_size, hidden_size]
    """
    batch_size, beam_size = row_id.size()
    if type(hidden) is tuple or type(hidden) is list:
        new_hidden = []
        for h in hidden:
            new_h = h.view(batch_size, beam_size, -1)[batch_id.data, row_id.data]
            new_h = new_h.view(batch_size * beam_size, -1)
            new_hidden.append(new_h)
        new_hidden = tuple(new_hidden)
    else:
        new_hidden = hidden.view(batch_size, beam_size, hidden.size(2))[:, batch_id.data, row_id.data]
        new_hidden = new_hidden.view(batch_size * beam_size, hidden.size(2))
    return new_hidden 
   

def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask:
        att_weight.data.masked_fill_(mask, -float('inf'))
    att_weight = F.softmax(att_weight)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight


class StackedAttentionLSTM(nn.Module):
    """
    stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedAttentionLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.att_vec_linear = nn.Linear(rnn_size * 3, rnn_size, bias=False)

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden, src_encoding, src_encoding_att_linear):
        """
        :param input: (batch_size, input_size)
        :param hidden : (num_layer, batch_size, hidden_size)
        :param src_encoding: (batch_size, src_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_len, hidden_size)
        return: input: (batch_size, hidden_size)
                h_1, c_1: (num_layers, batch_size, hidden_size)
        """
        h_0, c_0 = hidden
        h_1_0, c_1_0 = self.layers[0](input, (h_0[0], c_0[0]))
        h_1, c_1 = [h_1_0], [c_1_0]
        # Only use the first decoding outputs to do attention and copy the context vectors
        # to the subsequent decoding layers
        ctx_t, alpha_t = dot_prod_attention(h_1_0, src_encoding, src_encoding_att_linear)
        input = self.att_vec_linear(torch.cat([h_1_0, ctx_t], 1))  # (batch, hidden_size)

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = self.att_vec_linear(torch.cat([h_1_i, ctx_t], 1))
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        input = F.tanh(input)
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        h_1 = self.dropout(h_1)
        return input, (h_1, c_1)


class Decoder(nn.Module):
    """ Base class for decoder """
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
    
    def forward(self, src_hidden, final_src_ctx, trg_seq):
        """ 
        Args:
            src_hidden: [batch, src_len, encoder_hidden_size]
            final_src_ctx: tuple of (last_state, last_cell), [batch, encoder_hidden_size]
            trg_seq: [batch, trg_len]
        Return:
            scores: probability of words [batch, trg_len, vocab_size]
        """
        raise NotImplementedError
    
    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)  

    def init_state(self, final_src_ctx):
        print('using base init_state')
        raise NotImplementedError      
    
    def decode_one_step(self, input, hidden, src_hidden, ctx_vec, beam_size):
        """ Decode only one step
        Args: 
            input: [batch], the current input
            hidden: [batch, hidden_size], the previous hidden state
            ctx_vec: [batch, d], (optional) context vector to append to the input at each decoding step
            src_hidden: [batch, src_seq, D], (optional) for attention on source sequence
        Return:
            log_prob: [batch, num_vocab]
            hidden: [batch, hidden_size], next hidden state
        """
        raise NotImplementedError
    
    def beam_search(self, src_hidden, final_src_ctx, ctx_vec, repeat_hidden=True, repeat_ctx=False, beam_size=5, decode_max_length=100, to_word=True):
        """
        Args:
            src_hidden: [batch, src_seq_len, hidden_size]
            final_src_ctx: tensor or tuple of tensor [batch, hidden_size], used to initialize decoder hidden state
            ctx_vec: [batch, hidden_size], context vector that adds to the input at each decoding step
        Return:
            sequence: [batch, beam_size, decode_max_length] 
            sort_score: [batch, beam_size]
        """
        batch_size = src_hidden.size(0)
        num_vocab = len(self.vocab)   
        new_tensor = src_hidden.data.new
        pad_id = self.vocab['<pad>']
        eos_id = self.vocab['</s>']
        bos_id = self.vocab['<s>']

        # repeat `src_hidden`, `pad_mask` for k times
        dec_input = Variable(new_tensor(batch_size * beam_size).long().fill_(bos_id))  # [batch*beam_size]
        dec_hidden = self.init_state(final_src_ctx)  # [batch, hidden_size]

        if repeat_hidden:
            dec_hidden = repeat(dec_hidden, dim=1, k=beam_size)  # [batch*beam_size, hidden]

        if repeat_ctx:
            ctx_vec = repeat(ctx_vec, dim=1, k=beam_size)  # [batch*beam_size, hidden_size]

        batch_id = Variable(torch.LongTensor(range(batch_size))[:, None].repeat(1, beam_size).type_as(dec_input.data))
        
        top_score = Variable(new_tensor(batch_size, beam_size).fill_(-float('inf')))
        top_score.data[:, 0].fill_(0)

        # [1, |V|]
        pad_vec = new_tensor(1, num_vocab).float().fill_(-float('inf'))
        pad_vec[0, pad_id] = 0
        eos_mask = new_tensor(batch_size, beam_size).byte().fill_(0)  # [batch, beam_size]

        top_rowids, top_colids = [], []
        for i in range(decode_max_length):
            # [batch*beam_size, |V|]
            log_prob, dec_hidden, ctx_vec = self.decode_one_step(dec_input, dec_hidden, src_hidden, ctx_vec, beam_size)
            log_prob = log_prob.view(batch_size, beam_size, num_vocab)  # [batch, beam_size, |V|]
            
            if eos_mask is not None and eos_mask.sum() > 0:
                log_prob.data.masked_scatter_(eos_mask.unsqueeze(2), pad_vec.expand((eos_mask.sum(), num_vocab))) # [batch]
            
            score = top_score.unsqueeze(2) + log_prob   # [batch, beam_size, |V|]

            # [batch, beam_size]
            top_score, top_rowid, top_colid = topK_2d(score)  
            top_rowids.append(top_rowid)    # select the beam id
            top_colids.append(top_colid)    # select the word id

            dec_hidden = select_hid(dec_hidden, batch_id, top_rowid)  # [batch*beam_size, hidden_size]
            dec_input = top_colid.view(-1)  # [batch_size * beam_size]

            eos_mask = eos_mask.gather(dim=1, index=top_rowid.data) | top_colid.data.eq(eos_id)
            if eos_mask.sum() == batch_size * beam_size:
                break
        
        tokens = []
        for i in reversed(range(len(top_colids))):
            if i == len(top_colids) - 1:
                sort_score, sort_idx = torch.sort(top_score, dim=1, descending=True)  # [batch_size, beam_size]
                # sort_score, sort_idx = sort_score[:, :N], sort_idx[:, :N]
            else:
                sort_idx = top_rowids[i+1].gather(dim=1, index=sort_idx)
            token = top_colids[i].gather(dim=1, index=sort_idx)
            
            tokens.insert(0, token)
            # print('i', i, 'token', token.size)
        sequence = torch.stack(tokens, dim=2)  # [batch, beam_size, decode_max_lengths]
        if to_word:
            # print('sequence', sequence.size())
            sequence = sequence.cpu().data.numpy().tolist()
            sequence = [[[self.vocab.id2word[w] for w in seq if w != self.vocab['<pad>']] for seq in beam] for beam in sequence]
            # print('one sentence', sequence[0][0])
            sort_score = sort_score.cpu().data.numpy().tolist()
        #dec_hidden = dec_hidden.contiguous().view(batch_size, beam_size, hidden_size)
        return sequence, sort_score, dec_hidden

    # def sample(self, src_hidden, final_src_ctx, ctx_vec, repeat_hidden=True, repeat_ctx=False, sample_size=5, decode_max_length=100, to_word=True, sample_method='random', detached=False):
    #     """
    #     Args:
    #         src_hidden: [batch, src_seq_len, hidden_size]
    #         final_src_ctx: tensor or tuple of tensor [batch, hidden_size], used to initialize decoder hidden state
    #         ctx_vec: [batch, hidden_size], context vector that adds to the input at each decoding step
    #     Return:
    #         completed_sequences: [batch, sample_size, decode_max_length] 
    #         completed_scores: [batch, sample_size, decode_max_length]
    #         last_dec_hidden: [batch, sample_size, hidden_size]
    #     """
    #     batch_size =  int(src_hidden.size(0))
    #     aug_batch_size = batch_size * sample_size  # augmented batch size
    #     eos_id = self.vocab['</s>']
    #     bos_id = self.vocab['<s>']
        
    #     # Initial the first decoding input and the init state and cell
    #     new_tensor = src_hidden.data.new
    #     dec_input = Variable(new_tensor(aug_batch_size).long().fill_(bos_id))  # [batch*sample_size]
    #     dec_hidden = self.init_state(final_src_ctx)  # [batch, hidden_size]

    #     if repeat_hidden:
    #         dec_hidden = repeat(dec_hidden, dim=1, k=sample_size)  # [batch*sample_size, hidden_size]
        
    #     if repeat_ctx:
    #         ctx_vec = repeat(ctx_vec, dim=1, k=sample_size)  # [batch*sample_size, hidden_size]   

    #     # Repeat the source hidden states
    #     src_hidden_repeat = repeat(src_hidden, dim=1, k=sample_size)  # [batch*sample_size, hidden]

    #     # Ending conditions
    #     sample_ends = new_tensor([0] * aug_batch_size).byte()
    #     all_ones = new_tensor([1] * aug_batch_size).byte()
    #     # hyp_scores = Variable(new_tensor(aug_batch_size).float().zero_())

    #     samples = [dec_input]
    #     samples_losses = []
    #     samples_scores = []

    #     for t in range(decode_max_length):
    #         # Decoding one step and get the log-probability
    #         log_prob, dec_hidden, ctx_vec = self.decode_one_step(dec_input, dec_hidden, src_hidden, ctx_vec, sample_size)

    #         if sample_method == 'random':
    #             dec_input = torch.multinomial(F.softmax(log_prob), num_samples=1)
    #             sampled_log_prob = log_prob.gather(1, dec_input).squeeze(1) * Variable(1 - sample_ends).float()
    #             # hyp_scores += sampled_log_prob
    #             dec_input = dec_input.squeeze(1)

    #         if detached:
    #             dec_input = dec_input.detach()
    #         samples.append(dec_input)
    #         samples_scores.append(sampled_log_prob)

    #         # Check whether all samples are completed
    #         sample_ends |= torch.eq(dec_input, eos_id).byte().data
    #         if torch.equal(sample_ends, all_ones):
    #             break
    #     last_dec_hidden = resize(dec_hidden, (batch_size, sample_size, -1))
        
    #     # Post-process log-probability of each token at each decoding step
    #     completed_scores = [[[0] * decode_max_length] * sample_size] * batch_size
    #     for t, sample_scores in enumerate(samples_scores):
    #         for i, score in enumerate(sample_scores):
    #             src_sent_id = i % batch_size 
    #             sample_id = i // batch_size
    #             completed_scores[src_sent_id][sample_id][t] = score 

    #     # Post-process sampled sequence 
    #     masks = [list([list() for _ in range(sample_size)]) for _ in range(batch_size)]
    #     completed_sequences = [list([list() for _ in range(sample_size)]) for _ in range(batch_size)]
    #     for y_t in samples:
    #         for i, sampled_word in enumerate(y_t.cpu().data):
    #             src_sent_id = i % batch_size 
    #             sample_id = i // batch_size
    #             if len(completed_sequences[src_sent_id][sample_id]) == 0 or completed_sequences[src_sent_id][sample_id][-1] != eos_id:
    #                 completed_sequences[src_sent_id][sample_id].append(int(sampled_word))
    #                 masks[src_sent_id][sample_id].append(0)
    #             else:
    #                 masks[src_sent_id][sample_id].append(1)

    #     if to_word:
    #         for i, src_sent_samples in enumerate(completed_sequences):
    #             completed_sequences[i] = [[self.vocab.id2word[w] for w in s] for s in src_sent_samples]
    #     return completed_sequences, completed_scores, last_dec_hidden


class LSTMDecoder(Decoder):
    def __init__(
        self, vocab, embed_size=512, hidden_size=512, num_layers=1, 
        dropout=0.2, encoder_hidden_size=512, encoder_ctx_size=512
        ):
        super().__init__(vocab)
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Embedding lookup table
        self.trg_embed = nn.Embedding(len(vocab), embed_size, padding_idx=vocab['<pad>'])
        # decoder LSTMCell
        self.lstm = StackedAttentionLSTM(num_layers, embed_size + hidden_size, hidden_size, dropout=dropout)
        # Attention: project source hidden vectors to the decoder RNN's hidden space
        self.att_src_linear = nn.Linear(encoder_hidden_size, hidden_size, bias=False)
        # Initialize decoder hidden state
        self.decoder_init_linear = nn.Linear(encoder_ctx_size, hidden_size)
        # prediction layer of the target vocabulary
        self.readout = nn.Linear(hidden_size, len(vocab), bias=False)

    def init_state(self, final_src_ctx):
        last_state, last_cell = final_src_ctx    # [batch, encoder_hidden_size]
        #print('last_cell', last_cell.size())
        hsz = last_cell.size()
        # print('last_cell', last_cell.size())
        last_cell = last_cell.view(hsz[0], self.num_layers, -1)
        # print('last_cell', last_cell.size())
        init_cell = self.decoder_init_linear(last_cell)
        init_state = F.tanh(init_cell)
        init_cell = init_cell.view(hsz[0], -1)
        init_state = init_state.view(hsz[0], -1)

        # print('init_cell', init_cell.size())
        # print('end of init')
        return (init_state, init_cell)

    def forward(self, src_hidden, final_src_ctx, trg_seq):
        init_cell = self.decoder_init_linear(final_src_ctx[1])
        init_state = F.tanh(init_cell)
        # init_state, init_cell = self.init_state(final_src_ctx)
        init_state = init_state.unsqueeze(0).repeat(self.num_layers, 1, 1)   # [num_layers, batch_size, decoder_hidden_size] 
        init_cell = init_cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        hidden = (init_state, init_cell)
        batch_size = src_hidden.size(0)
        new_tensor = src_hidden.data.new

        # For attention, project source hidden vectors to decoder hidden dimension space
        src_hidden_att_linear = self.att_src_linear(src_hidden)   # [batch, src_len, hidden_size]
        # initialize attentional vector
        att_tm1 = Variable(new_tensor(batch_size, self.hidden_size).zero_(), requires_grad=False)

        trg_word_embed = self.trg_embed(trg_seq)  # [batch, trg_len, embed_size]
        scores = []
        # start from <s>, util y_{T-1}
        for y_tm1_embed in trg_word_embed.split(split_size=1, dim=1):
            # input feeding: concatenate y_{t-1} and previous attentional vector
            x = torch.cat([y_tm1_embed.squeeze(1), att_tm1], 1)
            # Go through LSTMCell to get h_t [batch_size, hidden_size]
            att_t, (h_t, cell_t) = self.lstm(x, hidden, src_hidden, src_hidden_att_linear)
            score_t = self.readout(att_t)   # [batch, |V|]
            scores.append(score_t)

            att_tm1 = att_t
            hidden = (h_t, cell_t)
        scores = torch.stack(scores, dim=1)  # [batch, seq_len, |V|]
        return scores

    def decode_one_step(self, input, hidden, src_hidden, ctx_vec, beam_size=5):
        y_tm1 = self.trg_embed(input)
        src_hidden = repeat(src_hidden, dim=1, k=beam_size)
        src_hidden_att_linear = self.att_src_linear(src_hidden)
        x = torch.cat([y_tm1, ctx_vec], dim=1)
        
        # For multi-layer LSTM
        # reshape [batch*beam_size, num_layers*hidden_size] to [num_layers, batch*beam_size, hidden_size]
        hsz = hidden[0].size()
        # print('h_t', hsz)
        h_t = hidden[0].view(hsz[0], self.num_layers, -1).transpose(0, 1)
        c_t = hidden[1].view(hsz[0], self.num_layers, -1).transpose(0, 1) # [num_layers, batch*beam_size, hidden_size]
        # print('h_t', h_t.size())
        att_t, (h_t, c_t) = self.lstm(x, (h_t, c_t), src_hidden, src_hidden_att_linear)
        score_t = self.readout(att_t)  # [num_seq, |V|]
        # print('h_z', h_t.size())
        # reshape back to [batch*beam_size, num_layers*hidden_size]
        h_t = h_t.transpose(0, 1).contiguous().view(hsz[0], -1)
        c_t = c_t.transpose(0, 1).contiguous().view(hsz[0], -1)
        # print('h_z', h_t.size())
        return F.log_softmax(score_t, dim=-1), (h_t, c_t), att_t

    def generate(self, src_hidden, final_src_ctx, beam_size=5, decode_max_length=100, to_word=True, length_norm=True):
        # initialize the first decoder hidden state
        batch_size = src_hidden.size(0)
        ctx_vec = Variable(src_hidden.data.new(src_hidden.size(0), self.hidden_size).zero_(), requires_grad=False)
        final_src_ctx = repeat(final_src_ctx, dim=1, k=beam_size)    # [batch*beam_size, enc_hidden_size]
        h_T, c_T = final_src_ctx
        h_T = h_T.unsqueeze(1).repeat(1, self.num_layers, 1).contiguous().view(batch_size * beam_size, -1)
        c_T = c_T.unsqueeze(1).repeat(1, self.num_layers, 1).contiguous().view(batch_size * beam_size, -1)
        final_src_ctx = (h_T, c_T)   # [batch*beam_size, num_layer * enc_hidden_size]

        completed_sequences, completed_scores, dec_hidden = self.beam_search(src_hidden, final_src_ctx, ctx_vec, False, True, beam_size, decode_max_length, to_word)
        if length_norm:
            completed_sequences, completed_scores = self.sort_by_len(completed_sequences, completed_scores)

        return completed_sequences, completed_scores

    def sort_by_len(self, seqs, scores):
        """
        Args:
            seqs: [num_seq, beam, seq_len]
            scores: [num_seq, beam]
        """
        new_seqs, new_scores = [], []
        for seq, score in zip(seqs, scores):
            for i in range(len(score)):
                score[i] /= len(seq[i])
            pairs = list(zip(score, seq))
            sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
            sorted_score, sorted_seq = zip(*sorted_pairs)
            new_seqs.append(sorted_seq)
            new_scores.append(sorted_score)
        return new_seqs, new_scores    