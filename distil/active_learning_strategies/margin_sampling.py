import numpy as np
from .strategy import Strategy
import pdb

class MarginSampling(Strategy):

   """
    Implementation of Margin Sampling Strategy.
    This class extends :class:`active_learning_strategies.strategy.Strategy`
    to include margin sampling technique to select data points for active learning.
    
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
        super(MarginSampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

    def select(self, budget):
    	"""
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        
        Returns
        ----------
        U_idx: list
            List of selected data point indexes with respect to unlabeled_x
        """
        probs = self.predict_prob(self.unlabeled_x)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        U_idx = U.sort()[1].numpy()[:budget] 
        return U_idx
