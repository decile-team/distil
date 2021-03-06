from .strategy import Strategy

class LeastConfidence(Strategy):
    """

    Implements the Least Confidence Sampling Strategy a active learning strategy where 
    the algorithm selects the data points for which the model has the lowest confidence while 
    predicting its label.
    
    Suppose the model has `nclasses` output nodes denoted by :math:`\\overrightarrow{\\boldsymbol{z}}` 
    and each output node is denoted by :math:`z_j`. Thus, :math:`j \\in [1, nclasses]`. 
    Then for a output node :math:`z_i` from the model, the corresponding softmax would be 

    .. math::
        \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}} 

    Then the softmax can be used pick `budget` no. of elements for which the model has the lowest 
    confidence as follows, 
    
    .. math::
        \\mbox{argmin}_{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\sum_S(\mbox{argmax}_j{(\\sigma(\\overrightarrow{\\boldsymbol{z}}))})}  
    

    where :math:`\\mathcal{U}` denotes the Data without lables i.e. `unlabeled_x` and :math:`k` is the `budget`.
    
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
