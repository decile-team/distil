import numpy as np
import torch
from .strategy import Strategy

class BALDDropout(Strategy):
	def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
		if 'n_drop' in args:
			self.n_drop = args['n_drop']
		else:
			self.n_drop = 10

		super(BALDDropout, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args={})

	def select(self, budget):

		probs = self.predict_prob_dropout_split(self.unlabeled_x, self.n_drop)
		pb = probs.mean(0)
		entropy1 = (-pb*torch.log(pb)).sum(1)
		entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
		U = entropy2 - entropy1
		idxs = U.sort()[1][:budget]

		return idxs
