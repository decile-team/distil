import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
    
    """

    Implements KMeans Sampling selection strategy, the last layer embeddings are calculated for 
    all the unlabeled data points.Then the KMeans clustering algorithm is run over these embeddings 
    with the number of clusters equal to the `budget`. Then the distance is calculated for all the 
    points from their respective centers. From each cluster, the point closest to the center is 
    selected to be labeled for the next iteration. Since the number of centers are equal to the 
    budget, selecting one point from each cluster satisfies the total number of data points to be 
    selected in one iteration.

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
    	super(KMeansSampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args={})

    def select(self, budget):
        
    	"""
        Select next set of points

        Parameters
        ----------
        budget: int
            Nuber of indexes to be returned for next set

        Returns
        ----------
        q_idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """

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