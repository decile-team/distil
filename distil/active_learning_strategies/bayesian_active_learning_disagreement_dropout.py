import torch
from .strategy import Strategy

class BALDDropout(Strategy):
    """
    Implementation of BALDDropout Strategy.
    This class extends :class:`active_learning_strategies.strategy.Strategy`
    to include entropy sampling technique to select data points for active learning.

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
        super(BALDDropout, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args={})

    def select(self, budget):

        """
        Select next set of points

        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set

        Returns
        ----------
        idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """	
		
        probs = self.predict_prob_dropout_split(self.unlabeled_x, self.n_drop)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        U = entropy2 - entropy1
        idxs = U.sort()[1][:budget]

        return idxs
