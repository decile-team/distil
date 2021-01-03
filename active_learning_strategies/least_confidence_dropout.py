import numpy as np
import torch
from .strategy import Strategy

class LeastConfidenceDropout(Strategy):
	def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, n_drop=10):
		self.n_drop = n_drop
		super(LeastConfidenceDropout, self).__init__(X, Y, unlabeled_x, net, handler, nclasses)

	def select(self, n):
		probs = self.predict_prob_dropout(self.unlabeled_x, self.n_drop)
		U = probs.max(1)[0]
		U_idx = U.sort()[1][:n]
		return U_idx
