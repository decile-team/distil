import numpy as np
import torch
from .strategy import Strategy

class MarginSamplingDropout(Strategy):
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
        super(MarginSamplingDropout, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

    def select(self, budget):

        probs = self.predict_prob_dropout(self.unlabeled_x, self.n_drop)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        U_idx = U.sort()[1][:budget]
        return U_idx