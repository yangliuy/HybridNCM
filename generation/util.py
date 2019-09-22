from collections import defaultdict
import numpy as np
import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import corpus_bleu
import argparse

# class LossRecorder(object):
#     def __init__(self, loss_list, factor_list, name=None):
#         self.name = name
#         self.loss_list = loss_list
#         self.factor_val = {l:f for (l, f) in zip(loss_list, factor_list)}
#         self.loss_val = {l:0.0 for l in loss_list}
#         self.num_val = {l:0.0 for l in loss_list}
#     def increment(self, val_list, num):
#         for l in val_list:
#             v = val_list[l]
#             if v != 0 and l in self.loss_val:
#                 self.loss_val[l] += v * self.factor_val[l] * num 
#                 self.num_val[l] += self.factor_val[l] * num
#     def reset(self, loss_list):
#         for l in loss_list:
#             self.loss_val[l] = 0.0
#             self.num_val[l] = 0.0
#     def log(self, loss_list):
#         return ', '.join(['%s=%.4f' % (l, self.loss_val[l]/self.num_val[l]) for l in loss_list if self.loss_val[l] > 0 and self.num_val[l] > 0])


def read_corpus(file_path, pad_bos_eos=False):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if pad_bos_eos:
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


# def read_bitext(file_path, delimiter="|||"):
#     """ Read parallel text with the format: 'src ||| trg' """
#     src_sents, trg_sents = [], []
#     for line in open(file_path):
#         src_trg = line.strip().split(delimiter)
#         src = src_trg[0].strip().split(' ')
#         trg = ['<s>'] + src_trg[1].strip().split(' ') + ['</s>']
#         src_sents.append(src)
#         trg_sents.append(trg)
#     return src_sents, trg_sents


# def batch_slice(data, batch_size, sort=True):
#     batch_num = int(np.ceil(len(data) / float(batch_size)))
#     for i in range(batch_num):
#         cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
#         src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
#         trg_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

#         if sort:
#             src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
#             src_sents = [src_sents[src_id] for src_id in src_ids]
#             trg_sents = [trg_sents[src_id] for src_id in src_ids]

#         yield src_sents, trg_sents


# def data_iter(data, batch_size, shuffle=True):
#     """
#     randomly permute data, then sort by source length, and partition into batches
#     ensure that the length of source sentences in each batch is decreasing
#     """

#     buckets = defaultdict(list)
#     for pair in data:
#         src_sent = pair[0]
#         buckets[len(src_sent)].append(pair)

#     batched_data = []
#     for src_len in buckets:
#         tuples = buckets[src_len]
#         if shuffle: np.random.shuffle(tuples)
#         batched_data.extend(list(batch_slice(tuples, batch_size)))

#     if shuffle:
#         np.random.shuffle(batched_data)
#     for batch in batched_data:
#         yield batch


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]

def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    
    new_sents, masks = [], []
    for s in sents:
        masks.append([1] * len(s) + [0] * (max_len - len(s)))
        new_sents.append(s + [pad_token] * (max_len - len(s)))
    return new_sents, masks

def to_input_variable(sents, vocab, cuda=False, is_test=False):
    """
    return a tensor of shape (batch_size, src_sent_len)
    """

    word_ids = word2id(sents, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


# def get_bleu(references, hypotheses):
#     # compute BLEU
#     bleu_score = corpus_bleu([[ref[1:-1]] for ref in references],
#                              [hyp[1:-1] for hyp in hypotheses])

#     return bleu_score


# def get_acc(references, hypotheses, acc_type='word_acc'):
#     assert acc_type == 'word_acc' or acc_type == 'sent_acc'
#     cum_acc = 0.

#     for ref, hyp in zip(references, hypotheses):
#         ref = ref[1:-1]
#         hyp = hyp[1:-1]
#         if acc_type == 'word_acc':
#             acc = len([1 for ref_w, hyp_w in zip(ref, hyp) if ref_w == hyp_w]) / float(len(hyp) + 1e-6)
#         else:
#             acc = 1. if all(ref_w == hyp_w for ref_w, hyp_w in zip(ref, hyp)) else 0.
#         cum_acc += acc

#     acc = cum_acc / len(hypotheses)
#     return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='eval', type=str, help='run mode')
    parser.add_argument('--ref', type=str, help='path to the reference file')
    parser.add_argument('--hyp', type=str, help='path to the hypotheses file')
    args = parser.parse_args()

    if args.mode == 'eval':
        hyp = read_corpus(args.hyp, source='trg')
        ref = read_corpus(args.ref, source='trg')
        bleu = get_bleu(ref, hyp)
        word_acc = get_acc(ref, hyp, acc_type='word_acc')
        sent_acc = get_acc(ref, hyp, acc_type='sent_acc')
        print('BLEU={}, Word Accuracy={}, Sentence Accuracy={}\n'.format(bleu, word_acc, sent_acc))
        

