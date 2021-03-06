import numpy as np
import torch
from .strategy import Strategy

class EntropySamplingDropout(Strategy):
    """
    Implements the Entropy Sampling Strategy with dropout. Entropy Sampling Strategy is one 
    of the most basic active learning strategies, where we select samples about which the model 
    is most uncertain. To quantify the uncertainity we use entropy and therefore select points 
    which have maximum entropy. 

    Suppose the model has `nclasses` output nodes and each output node is denoted by :math:`z_j`. Thus,  
    :math:`j \in [1,nclasses]`. Then for a output node :math:`z_i` from the model, the corresponding 
    softmax would be 

    .. math::
        \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}
    
    Then entropy can be calculated as,

    .. math:: 
        ENTROPY = -\\sum_j \\sigma(z_j)*log(\\sigma(z_i))

    The algorithm then selects `budget` no. of elements with highest **ENTROPY**.
    
    The drop out version uses the predict probability dropout function from the base strategy class to find the hypothesised labels.
    User can pass n_drop argument which denotes the number of times the probabilities will be calculated.
    The final probability is calculated by averaging probabilities obtained in all iteraitons.    

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

        n_drop
        Dropout value to be used (int, optional)
    """

    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        """
        Constructor method
        """
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
        super(EntropySamplingDropout, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)
        
        
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
        probs = self.predict_prob_dropout(self.unlabeled_x, self.n_drop)
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        U_idx = U.sort()[1][:budget]
        return U_idx
