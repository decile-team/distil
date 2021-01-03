import numpy as np
from .strategy import Strategy
import pdb

class MarginSampling(Strategy):
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        super(MarginSampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

    def select(self, n):

        probs = self.predict_prob(self.unlabeled_x)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        U_idx = U.sort()[1].numpy()[:n] 
        print(U_idx)
        return U_idx
