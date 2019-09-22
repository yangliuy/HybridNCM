from encoder import Encoder, LSTMEncoder
from decoder import Decoder, LSTMDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from vocab import Vocab

def masked_lstm_encode(sent_emb, sent_len, lstm, batch_first=False):
    """
    Args:
        sent_emb: [len, batch, d] or [batch, len, d] if batch_first=True
        sent_len: numpy array or list (batch,)
        lstm: nn.LSTM encoder
    Return:
        output: [len, batch, D/2D] or [batch, len, D/2D]
        (last_state, last_cell): [num_layers, batch, D]
    """
    sidx = sorted(range(len(sent_len)), key=lambda x: sent_len[x], reverse=True)
    sent_len_sort = np.array(sent_len)[sidx]
    ridx = np.argsort(sidx)
    new_tensor = sent_emb.data.new
    sidx = new_tensor(sidx).long()
    ridx = new_tensor(ridx).long()
    sent_emb_sort = sent_emb[sidx]
    packed_sent_emb = pack_padded_sequence(sent_emb_sort, sent_len_sort, batch_first=batch_first)  
    sent_hid_sort, (last_state, last_cell) = lstm(packed_sent_emb)
    output, _ = pad_packed_sequence(sent_hid_sort)   # [len, batch, D or 2D] uni-/bi-directional LSTM
    if batch_first:
        output = output.transpose(0, 1)
    output = output[ridx].contiguous()
    last_state = last_state[:, ridx]  # [num_layers, batch, D]
    last_cell = last_cell[:, ridx]
    return output, (last_state, last_cell)

class HNCMEncoder(LSTMEncoder):
    def __init__(self, vocab, embed_size, hidden_size, num_layers, dropout, bidirectional=True, fusion_type='average'):
        print('in', vocab, embed_size, hidden_size, num_layers, dropout, bidirectional)
        # x = super()
        # print('x', x)
        super().__init__(vocab, embed_size, hidden_size, num_layers, dropout, bidirectional)
        self.fusion_type = fusion_type
        self.fact_encoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, src_seq, src_lengths, fact_seq, fact_lengths):
        # Context encoding
        src_embed = self.embed(src_seq)                 # [batch, src_len, embed_size]
        packed_src_embed = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_hidden, last_hidden = self.lstm(packed_src_embed)  # [batch, src_len, hidden_size*2]
        src_hidden, _ = pad_packed_sequence(src_hidden, batch_first=True)
        batch_size = src_hidden.size(0)
        h_T = last_hidden[0].transpose(0, 1).contiguous().view(batch_size, -1)  # [batch, num_layers * 2 * hidden_size]
        c_T = last_hidden[1].transpose(0, 1).contiguous().view(batch_size, -1)
        last_hidden = (h_T, c_T)

        # Fact encoding
        new_tensor = fact_seq.data.new
        batch_size, num_fact, max_fact_len = fact_seq.size()
        fact_embed = self.embed(fact_seq) # [batch, num_fact, max_fact_len, embed_size]
        fact_embed = fact_embed.view(batch_size*num_fact, max_fact_len, -1)
        fact_lengths_tensor = new_tensor(fact_lengths).long()  # [batch, num_fact]
        new_fact_lengths = fact_lengths_tensor.view(batch_size*num_fact).tolist()
        fact_hid, fact_last = masked_lstm_encode(fact_embed, new_fact_lengths, self.fact_encoder, batch_first=True)
        fact_hid = fact_hid.view(batch_size, num_fact, max_fact_len, -1)  # [batch, num_fact, max_fact_len, hidden_size]
        h_fact_T = fact_last[0].transpose(0, 1).contiguous().view(batch_size, num_fact, -1)
        c_fact_T = fact_last[1].transpose(0, 1).contiguous().view(batch_size, num_fact, -1)

        # Fusion of context and facts
        # Compute the mean vector of each fact
        # Concatnate the fact representation with source representation
        avg_fact_hid = fact_hid.sum(dim=2) / Variable(fact_lengths_tensor).float().unsqueeze(2)   # [batch, num_fact, hidden_size]
        avg_h_fact_T = h_fact_T.mean(dim=1)   # [batch, hidden_size]
        avg_c_fact_T = c_fact_T.mean(dim=1)

        if self.fusion_type == 'average': # use the average vector of word embeddings to represent a fact
            fusion_hid = torch.cat([src_hidden, avg_fact_hid], dim=1)  # [batch, num_src_len + num_fact, hidden_size]
        else: # use the last hidden state to represent a fact
            h_fact_T0 = fact_last[0][-2:].transpose(0,1).contiguous().view(batch_size, num_fact, -1)
            fusion_hid = torch.cat([src_hidden, h_fact_T0], dim=1) # [batch, num_src_len + num_fact, hidden_size]
        fusion_h_T = h_T + avg_h_fact_T
        fusion_c_T = c_T + avg_c_fact_T  
        
        return fusion_hid, (fusion_h_T, fusion_c_T)

class HNCMModel(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()

        self.args = args 
        self.encoder = encoder
        self.decoder = decoder 
        assert isinstance(self.encoder, Encoder)
        assert isinstance(self.decoder, Decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--embed_size', default=512, type=int)
        parser.add_argument('--hidden_size', default=512, type=int)
        parser.add_argument('--num_encoder_layers', default=1, type=int)
        parser.add_argument('--num_decoder_layers', default=1, type=int)
        parser.add_argument('--dropout', default=0.3, type=int)
        parser.add_argument('--bidirectional', action='store_true', default=False)

        # arguments which are only associated with HNCM model
        parser.add_argument('--train_fact_file', default=None, type=str, help='train fact data')
        parser.add_argument('--dev_fact_file', default=None, type=str, help='dev fact data')
        parser.add_argument('--test_fact_file', default=None, type=str, help='test fact data')
        parser.add_argument('--fact_num', default=None, type=str, help='number of facts in train/valid/test data')
        parser.add_argument('--fusion_type', default='average', type=str, help='fusion type')
        return parser

    # define forward process of hncm model and override the forward function in BaseModel
    def forward(self, src_seq, src_lengths, trg_seq, fact_seq, fact_lengths):
        fusion_hidden, final_fusion_ctx = self.encoder(src_seq, src_lengths, fact_seq, fact_lengths)
        scores = self.decoder(fusion_hidden, final_fusion_ctx, trg_seq)
        return scores

    def uniform_init(self):
        print('uniformly initialize parameters [-%f, +%f]' % (self.args.uniform_init, self.args.uniform_init))
        for name, param in self.named_parameters():
            print(name, param.size())
            param.data.uniform_(-self.args.uniform_init, self.args.uniform_init)

    def save(self, path):
        params = {
            'args': self.args,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_vocab': self.encoder.vocab,
            'decoder_vocab': self.decoder.vocab
        }
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=lambda storage, loc: storage)
        args = params['args']
        vocab = Vocab()
        vocab.src = params['encoder_vocab']
        vocab.trg = params['decoder_vocab']
        model = self.build_model(args, vocab)
        model.encoder.load_state_dict(params['encoder_state_dict'])
        model.decoder.load_state_dict(params['decoder_state_dict'])
        return model 

    @staticmethod
    def build_model(args, vocab):
        print('build HNCMModel (Hybrid Neural Conversational Model)...')
        encoder_hidden_size = 2 * args.hidden_size if args.bidirectional else args.hidden_size
        encoder_ctx_size = encoder_hidden_size * args.num_encoder_layers
        print('vocab', len(vocab.src), args.embed_size, args.hidden_size, args.num_encoder_layers, args.dropout)
        encoder = HNCMEncoder(vocab.src, args.embed_size, args.hidden_size, args.num_encoder_layers, args.dropout, bidirectional=args.bidirectional, fusion_type=args.fusion_type) # fusion_type='average'
        decoder = LSTMDecoder(vocab.trg, args.embed_size, args.hidden_size, args.num_decoder_layers, args.dropout,
                              encoder_hidden_size, encoder_ctx_size)
        return HNCMModel(args, encoder, decoder)

    def generate(self, src_seq, src_lengths, fact_seq, fact_lengths, beam_size=5, decode_max_length=100, to_word=True):
        src_hidden, final_src_ctx = self.encoder(src_seq, src_lengths, fact_seq, fact_lengths) # [batch, src_len, enc_hidden_size]

        completed_sequences, completed_scores = self.decoder.generate(src_hidden, final_src_ctx, beam_size,
                                                                      decode_max_length, to_word)
        return completed_sequences, completed_scores





