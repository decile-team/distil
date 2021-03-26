import numpy as np
from .strategy import Strategy
import pdb

class MarginSampling(Strategy):

    """
    Implementation of Margin Sampling Strategy.
    This class extends :class:`active_learning_strategies.strategy.Strategy`
    to include margin sampling technique to select data points for active learning.

    While least confidence only takes into consideration the maximum probability, margin sampling considers the difference between the confidence of first and the second most probable labels.  
  
    
    .. list-table:: Example
       :widths: 25 25 25 25
       :header-rows: 1

       * - Data Instances
         - Label 1
         - Label 2
         - Label 3
       * - p1
         - 0.1
         - 0.55
         - 0.45
       * - p2
         - 0.2
         - 0.3
         - 0.5
       * - p3
         - 0.1
         - 0.1
         - 0.8

    
    From the above table, the difference between the probability first and the second labels for p1, p2, p3 are 0.1, 0.2, 0.7 respectively.
    The margin sampling will query the true label for the data instance p1 since it has the smallest difference among all the different data instances. 

    Let :math:`p_i` represent probability for ith label and let there be n possible labels for data instance p. 
    Let :math:`\\max{(t)}` represent the maximum value in t and :math:`max1{(t)}` represent second maximum value in t then, mathematically it can be written as:
    
    
    .. math::
        \\min{(\\max{(P)} - \\max1{(P)})} 

    
    where P=[ :math:`p_1, p_2,… p_n`]
   
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
