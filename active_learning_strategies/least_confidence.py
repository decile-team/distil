import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses):
        super(LeastConfidence, self).__init__(X, Y, unlabeled_x, net, handler, nclasses)

    def select(self, n):

        probs = self.predict_prob(self.unlabeled_x)
        U = probs.max(1)[0]
        U_idx = U.sort()[1][:n]
        return U_idx
