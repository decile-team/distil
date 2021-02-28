import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
	def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
		super(KMeansSampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args={})

	def select(self, budget):

		embedding = self.get_embedding(self.unlabeled_x)
		embedding = embedding.numpy()
		cluster_learner = KMeans(n_clusters=budget)
		cluster_learner.fit(embedding)
		
		cluster_idxs = cluster_learner.predict(embedding)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (embedding - centers)**2
		dis = dis.sum(axis=1)
		q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(budget)])

		return q_idxs