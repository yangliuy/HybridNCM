from __future__ import print_function
import torch
import torch.nn as nn
from options import get_training_parser, parse_args
from hncm_trainer import HNCMTrainer
from hncm_dataloader import HNCMDataLoader
from hncm_model import  HNCMModel
import time
import numpy as np
import os
import sys
from datetime import datetime


def train(args):
    # load data
    data = HNCMDataLoader(args)
    vocab = data.vocab

    # initialize model
    model = HNCMModel.build_model(args, vocab)
    print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init)) # randomly init from U([-0.1, 0.1])
    model.uniform_init()
    # initialize loss
    vocab_mask = torch.ones(len(vocab.trg))
    vocab_mask[vocab.trg['<pad>']] = 0
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)
    # initialize Trainer
    trainer = HNCMTrainer(args, model, cross_entropy_loss)
    # logging

    print('Begining training')
    train_iter = valid_num = 0
    begin_time = time.time()
    train_loss = cum_train_loss = trg_word = cum_trg_word = train_example = cum_train_example = 0.0
    hist_valid_scores = []
    for epoch in range(args.max_epoch):
        for sample in HNCMDataLoader.data_iter(data.trn, vocab, batch_size=args.batch_size, shuffle=True, cuda=args.cuda):
            train_iter += 1
            loss, log_outputs = trainer.train_step(sample)

            train_loss += loss.item() * sample['num_trg_seq']  # number of sequences in target data
            cum_train_loss += loss.item() * sample['num_trg_seq']
            trg_word += sample['num_trg_word']
            cum_trg_word += sample['num_trg_word']  # number of words in target data
            train_example += sample['num_trg_seq']
            cum_train_example += sample['num_trg_seq']

            if train_iter % args.log_interval == 0:
                # print('train loss %.2f, train trg words %.2f' % (train_loss, trg_word))
                print('time %s, epoch %d, iter %d, avg. loss %.6f, avg. ppl %.2f ' \
                      'example %d, time elapsed %.2f seconds'
                      % (str(datetime.now()), epoch, train_iter,
                         train_loss / trg_word,
                         np.exp(train_loss / trg_word),
                         train_example,
                         time.time() - begin_time))
                # Reset the training log
                train_loss = trg_word = train_example = 0.0

                # Perform validation
            if train_iter % args.valid_interval == 0:
                print('time %s, epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f ' \
                      'time elapsed %.2f seconds' % (str(datetime.now()), epoch, train_iter,
                                                     cum_train_loss / cum_train_example,
                                                     np.exp(cum_train_loss / cum_trg_word),
                                                     time.time() - begin_time))
                cum_train_loss = cum_trg_word = cum_train_example = 0.0
                print('Begin validation ...')
                valid_num += 1
                dev_loss, dev_ppl = validation(args, vocab, trainer, data)
                print('time %s, Validation: epoch %d, iter %d, dev. loss %.2f, dev. ppl %.2f' % (
                str(datetime.now()), epoch, train_iter, dev_loss, dev_ppl))
                dev_metric = -dev_loss

                is_better = len(hist_valid_scores) == 0 or dev_metric > max(hist_valid_scores)
                is_better_than_last = len(hist_valid_scores) == 0 or dev_metric > hist_valid_scores[-1]
                hist_valid_scores.append(dev_metric)

                if valid_num > args.save_model_after:
                    model_file = args.save_model_to + '.iter%d.bin' % train_iter
                    print('save model to [%s]' % model_file)
                    model.save(model_file)

                if (not is_better_than_last) and args.lr_decay:
                    lr = trainer.optimizer.param_groups[0]['lr'] * args.lr_decay
                    print('decay learning rate to %f' % lr)
                    trainer.optimizer.param_groups[0]['lr'] = lr

                if is_better:  # If the validataion performances is better, reset the patience counter
                    patience = 0
                    best_model_iter = train_iter
                    if valid_num > args.save_model_after:
                        print('save the current best model ..')
                        model_file_abs_path = os.path.abspath(model_file)
                        symlin_file_abs_path = os.path.abspath(args.save_model_to + '.bin')
                        os.system('ln -sf %s %s' % (model_file_abs_path, symlin_file_abs_path))
                else:
                    patience += 1  # If the validataion performances didn't increase in adjacant patience times, the training will stop.
                    print('hit patience %d ' % patience)
                    if patience == args.patience:
                        print('early stop!')
                        print('the best model is from iteration [%d] ' % best_model_iter)
                        exit(0)


def validation(args, vocab, trainer, data):
    cum_loss = cum_trg_word = cum_example = 0.0
    for sample in HNCMDataLoader.data_iter(data.dev, vocab, batch_size=args.batch_size, shuffle=False, cuda=args.cuda):
        loss, log_outputs = trainer.valid_step(sample)
        cum_loss += loss.item() * sample['num_trg_seq']
        cum_trg_word += sample['num_trg_word']
        cum_example += sample['num_trg_seq']
    return cum_loss / cum_example, np.exp(cum_loss / cum_trg_word)


if __name__ == '__main__':
    parser = get_training_parser()
    parser = HNCMDataLoader.add_args(parser)
    parser = HNCMModel.add_args(parser)
    args = parser.parse_args()
    print('args', args)

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    train(args)
