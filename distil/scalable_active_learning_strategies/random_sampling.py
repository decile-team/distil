import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(RandomSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
    def select(self, budget):

        rand_idx = np.random.permutation(len(self.unlabeled_dataset))[:budget]
        rand_idx = rand_idx.tolist()
        return rand_idx