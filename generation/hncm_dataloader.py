import torch
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from vocab import Vocab
import os

def read_corpus(file_path, pad_bos_eos=False):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if pad_bos_eos:
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def read_fact(filename):
    facts = []
    for line in open(filename):
        fact = [f.strip().split(' ') for f in line.split(' tabplaceholder ')[0:-1]] # the last fact is empty
        facts.append(fact)
    return facts # 3D list of facts words

def word2id_2d(sents, vocab):  # batch * fact_num * fact_seq_len
    word_ids = []
    for fact_set in sents:
        fact_ids =  [[vocab[w] for w in s] for s in fact_set]
        word_ids.append(fact_ids)
    return word_ids

def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]

def pad_sentences(sents, pad_token):
    max_len = max(len(s) for s in sents)
    new_sents, masks = [], []
    for s in sents:
        masks.append([1] * len(s) + [0] * (max_len - len(s)))
        new_sents.append(s + [pad_token] * (max_len - len(s)))
    return new_sents, masks

def pad_sentences_2d(sents, pad_token):  # batch * fact_num * fact_word_num
    max_len = max(max(len(f) for f in facts) for facts in sents)
    new_sents, masks = [], []
    for fact_set in sents:
        new_sent, mask = [], []
        for s in fact_set:
            mask.append([1] * len(s) + [0] * (max_len - len(s)))
            new_sent.append(s + [pad_token] * (max_len - len(s)))
        new_sents.append(new_sent)
        masks.append(mask)
    return new_sents, masks

def to_input_var_2d(sents, vocab, cuda=False): # batch * fact_num * fact_word_num
    sents_id = word2id_2d(sents, vocab)
    sents_id, masks = pad_sentences_2d(sents_id, vocab.pad_id)
    sents_var = Variable(torch.LongTensor(sents_id), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()
    return sents_var

def to_input_var(sents, vocab, cuda=False):
    sents_id = word2id(sents, vocab)
    sents_id, masks = pad_sentences(sents_id, vocab.pad_id)
    sents_var = Variable(torch.LongTensor(sents_id), requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()
    return sents_var

class HNCMDataLoader(object):
    def __init__(self, args, vocab=None):
        self.args = args
        self.vocab = vocab 
        self.load_data()
        self.load_vocab()

    @staticmethod
    def add_args(parser):
        return parser

    def load_data(self):
        args = self.args
        def _load(src_file, trg_file, fact_file, delimiter='\t'):
            if  src_file is not None and trg_file is not None and fact_file is not None:
                src_sents = read_corpus(src_file, pad_bos_eos=False)
                trg_sents = read_corpus(trg_file, pad_bos_eos=True)
                fact_sents = read_fact(fact_file)
            else:
                src_sents = trg_sents = fact_sents = []
            return list(zip(src_sents, trg_sents, fact_sents)) #[src, trg, fact]
        self.trn = _load(args.train_src_file, args.train_trg_file, args.train_fact_file, args.delimiter)
        self.dev = _load(args.dev_src_file, args.dev_trg_file, args.dev_fact_file, args.delimiter)
        self.tst = _load(args.test_src_file, args.test_trg_file, args.test_fact_file, args.delimiter)

    def load_vocab(self):
        # Load the vocabulary or create vocabulary if not exists
        if self.args.vocab is not None:
            if not os.path.isfile(self.args.vocab):
                print('create new vocab and save to %s' % self.args.vocab)
                src_sents, trg_sents, fact_sents = zip(*self.trn)
                # when building vocab, treat fact sentences as additional source sentences
                # src_fact_sents contains a list of words in both source sentence and fact sentences
                src_fact_sents = []
                for src, fact in zip(src_sents, fact_sents):
                    new_fact = src
                    for f in fact:
                        new_fact += f
                    src_fact_sents.append(new_fact)
                self.vocab = Vocab(
                    src_fact_sents, trg_sents, self.args.src_vocab_size,
                    self.args.trg_vocab_size,
                    remove_singleton=not self.args.include_singleton,
                    share_vocab=self.args.share_vocab
                )
                torch.save(self.vocab, self.args.vocab)
            else:
                self.vocab = torch.load(self.args.vocab)
        else:
            print('vocab file is required')
            exit(0)

    @staticmethod
    def batch_slice(data, batch_size, sort=True):
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for i in range(batch_num):
            cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
            src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
            trg_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]
            fact_sents = [data[i * batch_size + b][2] for b in range(cur_batch_size)]

            if sort:
                src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
                src_sents = [src_sents[src_id] for src_id in src_ids]
                trg_sents = [trg_sents[src_id] for src_id in src_ids]
                fact_sents = [fact_sents[src_id] for src_id in src_ids]

            yield src_sents, trg_sents, fact_sents

    @staticmethod
    def data_iter(data, vocab, batch_size, shuffle=True, cuda=False):
        """
        Given data, generate sample for the current batch
        randomly permute data, then sort by source length, and partition into batches
        ensure that the length of source sentences in each batch is decreasing
        """

        buckets = defaultdict(list)
        for pair in data:
            src_sent = pair[0]
            buckets[len(src_sent)].append(pair)

        batched_data = []
        for src_len in buckets:
            tuples = buckets[src_len]
            if shuffle: np.random.shuffle(tuples)
            batched_data.extend(list(HNCMDataLoader.batch_slice(tuples, batch_size)))

        if shuffle:
            np.random.shuffle(batched_data)
        for src_sents, trg_sents, fact_sents in batched_data:
            num_trg_word = sum(len(s[:-1]) for s in trg_sents)
            src_lengths = [len(s) for s in src_sents]
            src_seqs_var = to_input_var(src_sents, vocab.src, cuda)
            trg_seqs_var = to_input_var(trg_sents, vocab.trg, cuda)
            fact_lengths = [[len (s) for s in fact_sent] for fact_sent in fact_sents]
            fact_seqs_var = to_input_var_2d(fact_sents, vocab.src, cuda)

            yield {
                'src_seq': src_seqs_var, 'src_lengths': src_lengths,
                'fact_seq': fact_seqs_var, 'fact_lengths': fact_lengths,
                'trg_seq': trg_seqs_var[:, :-1],
                'target': trg_seqs_var[:, 1:],
                'num_trg_word': num_trg_word, 'num_trg_seq': len(trg_sents)
            }
