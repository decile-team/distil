import numpy as np
import torch
from .strategy import Strategy

class MarginSamplingDropout(Strategy):
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, n_drop=10):
        self.n_drop = n_drop
        super(MarginSamplingDropout, self).__init__(X, Y, unlabeled_x, net, handler, nclasses)

    def select(self, n):

        probs = self.predict_prob_dropout(self.unlabeled_x, self.n_drop)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        U_idx = U.sort()[1][:n]
        return U_idx