from .strategy import Strategy

class MarginSamplingDropout(Strategy):
    """
    Implements the Margin Sampling Strategy with dropout a active learning strategy similar to Least Confidence 
    Sampling Strategy with dropout. While least confidence only takes into consideration the maximum probability, 
    margin sampling considers the difference between the confidence of first and the second most 
    probable labels.  
    
    Suppose the model has `nclasses` output nodes denoted by :math:`\\overrightarrow{\\boldsymbol{z}}` 
    and each output node is denoted by :math:`z_j`. Thus, :math:`j \\in [1, nclasses]`. 
    Then for a output node :math:`z_i` from the model, the corresponding softmax would be 

    .. math::
        \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}} 

    Let,

    .. math::
        m = \\mbox{argmax}_j{(\\sigma(\\overrightarrow{\\boldsymbol{z}}))}
        
    Then using softmax, Margin Sampling Strategy would pick `budget` no. of elements as follows, 
    
    .. math::
        \\mbox{argmin}_{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\sum_S(\\mbox{argmax}_j {(\\sigma(\\overrightarrow{\\boldsymbol{z}}))}) - (\\mbox{argmax}_{j \\ne m} {(\\sigma(\\overrightarrow{\\boldsymbol{z}}))})}  
    

    where :math:`\\mathcal{U}` denotes the Data without lables i.e. `unlabeled_x` and :math:`k` is the `budget`.
    
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
        super(MarginSamplingDropout, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

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
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        U_idx = U.sort()[1][:budget]
        return U_idx