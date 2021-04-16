import torch
from .strategy import Strategy

class EntropySampling(Strategy):
    """
    Implementation of Entropy Sampling Strategy.
    This class extends :class:`active_learning_strategies.strategy.Strategy`
    to include entropy sampling technique to select data points for active learning.

    Least Confidence and Margin Sampling do not make use of all the label probabilities, whereas entropy sampling calculates entropy based on the hypothesised confidence scores for each label and queries for the true label of a data instance with the highest entropy.
    

    .. list-table:: Example
       :widths: 50 50
       :header-rows: 1

       * - Data Instances
         - Entropy
       * - p1
         - 0.2
       * - p2
         - 0.5
       * - p3
         - 0.7


    From the above table, Entropy sampling will query for the true label data instance p3 since it has the highest entropy.

    Let :math:`p_i`  denote probability for ith label of data instance p, and let total possible labels be denoted by n, then Entropy for p is calculated as:
    

    .. math::
        E = \\sum{p_i*log(p_i)}
    
   
    where i=1,2,3....n   
    Thus Entropy Selection can be mathematically shown as:


    ..math::    
        \\max{(E)}     
    
    
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

        super(EntropySampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

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
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        U_idx = U.sort()[1][:budget]

        return U_idx
