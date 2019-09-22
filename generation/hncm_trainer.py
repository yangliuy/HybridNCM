from __future__ import print_function
import math
import torch
import numpy as np

class HNCMTrainer(object):
    def __init__(self, args, model, criterion):
        self.args = args

        self.model = model
        self.criterion = criterion

        # initialize optimizer and learning rate
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        # initialize loger
        self._num_updates = 0
        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def train_step(self, sample, objective='MLE'):
        # forward pass
        loss, log_outputs = self._forward(sample)
        # backward pass
        grad_norm = self._backward(loss)
        return loss, log_outputs

    def _forward(self, sample, eval=False):
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        # Forward pass to predict the log-probability
        lprobs = self.model(sample['src_seq'], sample['src_lengths'], sample['trg_seq'], sample['fact_seq'], sample['fact_lengths'])
        target = sample['target']

        # get the loss
        loss = self.criterion(lprobs.contiguous().view(-1, lprobs.size(-1)), target.contiguous().view(-1))
        loss = loss / sample['num_trg_seq']
        logging_outputs = {'loss': loss, 'nsample': sample['target'].size(0)}
        return loss, logging_outputs

    def _backward(self, loss):
    	loss.backward()

    	if self.args.clip_norm > 0:
    		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
    	else:
            grad_norm = math.sqrt(sum(p.grad.data.norm()**2 for p in self.model.parameters()))
    	self.optimizer.step()
    	self._num_updates += 1
    	return grad_norm

    def valid_step(self, sample):
        loss, log_outputs = self._forward(sample, eval=True)
        return loss, log_outputs
