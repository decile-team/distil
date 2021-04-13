from .strategy import Strategy

class LeastConfidence(Strategy):
    """
    Implementation of Least Confidence Sampling Strategy.
    This class extends :class:`active_learning_strategies.strategy.Strategy` to include least confidence technique to select data points for active learning.
    
    In this active learning strategy, the algorithm selects the data points for which the model has the lowest confidence while predicting its hypothesised label.
    
    
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

    
    From the above table, the label for instance p1 is 2 with a confidence of 0.55, for instance p2, the hypothesised label predicted is 3 with confidence of 0.5 and for p3 label 3 is predicted with a confidence of 0.8. Thus, according to least confidence strategy,  the point for which it will query for true label will be instance p2.

    Let :math:`p_i` represent probability for ith label and let there be n possible labels for data instance p then, mathematically it can be written as:
    
    
    .. math::
        \\min{(\\max{(P)})}  
    

    where P=:math:`[p_1, p_2,… p_n]`
    
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
        super(LeastConfidence, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

    def select(self, budget):
        """
        Select next set of points

        Parameters
        ----------
        budget: int
            Nuber of indexes to be returned for next set

        Returns
        ----------
        U_idx: list
            List of selected data point indexes with respect to unlabeled_x
        """

        probs = self.predict_prob(self.unlabeled_x)
        U = probs.max(1)[0]
        U_idx = U.sort()[1][:budget]
        return U_idx
