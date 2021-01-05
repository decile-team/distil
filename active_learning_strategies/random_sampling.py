import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        super(RandomSampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

    def select(self, budget):
        rand_idx = np.random.permutation(self.unlabeled_x.shape[0])[:budget]
        rand_idx = rand_idx.tolist()
        return rand_idx
