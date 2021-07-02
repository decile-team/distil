import numpy as np
from .strategy import Strategy
from sklearn.metrics import pairwise_distances

class CoreSet(Strategy):
    """
    Implementation of CoreSet :footcite:`sener2018active` Strategy. A diversity-based 
    approach using coreset selection. The embedding of each example is computed by the network’s 
    penultimate layer and the samples at each round are selected using a greedy furthest-first 
    traversal conditioned on all labeled examples.

    Parameters
    ----------
    X: numpy array
        Present training/labeled data   
    Y: numpy array
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

        if 'tor' in args:
            self.tor = args['tor']
        else:
            self.tor = 1e-4

        super(CoreSet, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

    def furthest_first(self, X, X_set, n):
        """
        Selects points with maximum distance
        
        Parameters
        ----------
        X: numpy array
            Embeddings of unlabeled set
        X_set: numpy array
            Embeddings of labeled set
        n: int
            Number of points to return
        Returns
        ----------
        idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """ 
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def select(self, budget):
        """
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        
        Returns
        ----------
        chosen: list
            List of selected data point indexes with respect to unlabeled_x
        """ 
        embedding_unlabeled = self.get_embedding(self.unlabeled_x)
        embedding_unlabeled = embedding_unlabeled.numpy()
        embedding_labeled = self.get_embedding(self.X)
        embedding_labeled = embedding_labeled.numpy()

        chosen = self.furthest_first(embedding_unlabeled, embedding_labeled, budget)
        return chosen