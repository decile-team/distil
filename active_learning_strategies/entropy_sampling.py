import numpy as np
import torch
from .strategy import Strategy

class EntropySampling(Strategy):
	def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
		super(EntropySampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

	def select(self, n):
		
		probs = self.predict_prob(self.unlabeled_x)
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		U_idx = U.sort()[1][:n]

		return U_idx
