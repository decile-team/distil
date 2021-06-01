import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):

    """
    Implementation of Random Sampling Strategy. This strategy is often used as a baseline, 
    where we pick a set of unlabeled points randomly.

    Parameters
    ----------
    X: numpy array
        Present training/labeled data   
    y: numpy array
        Labels of present training data
    unlabeled_x: numpy array
        Data without labels
    net: class
        Pytorch Model class
    handler: class
        Data Handler, which can load data even without labels.
    nclasses: int
        Number of unique target variables
    args: dict
        Specify optional parameters
        
        batch_size 
        Batch size to be used inside strategy class (int, optional)
    """

    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        """
        Constructor method
        """
        super(RandomSampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)
        
    def select(self, budget):
        """
        Select next set of points

        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set

        Returns
        ----------
        rand_idx: list
            List of selected data point indexes with respect to unlabeled_x
        """
        rand_idx = np.random.permutation(self.unlabeled_x.shape[0])[:budget]
        rand_idx = rand_idx.tolist()
        return rand_idx
