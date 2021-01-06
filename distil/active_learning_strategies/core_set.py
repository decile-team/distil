import numpy as np
import pdb
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
from sklearn.metrics import pairwise_distances

class CoreSet(Strategy):
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):

        if 'tor' in args:
            self.tor = args['tor']
        else:
            self.tor = 1e-4

        super(CoreSet, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

    def furthest_first(self, X, X_set, n):
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

        embedding_unlabeled = self.get_embedding(self.unlabeled_x)
        embedding_unlabeled = embedding_unlabeled.numpy()
        embedding_labeled = self.get_embedding(self.X)
        embedding_labeled = embedding_labeled.numpy()

        chosen = self.furthest_first(embedding_unlabeled, embedding_labeled, budget)
        print(chosen)

        if len(list(set(chosen))) < 10:
            print(embedding_unlabeled)
            print('Weights lm1')
            print(self.model.lm1.weight.data.numpy())
            print('Weights lm2')
            print(self.model.lm2.weight.data.numpy())
            print('Input X')
            print(self.unlabeled_x)

        return chosen