from .score_streaming_strategy import ScoreStreamingStrategy

class LeastConfidenceSamplingDropout(ScoreStreamingStrategy):
    
    """
    Implements the Least Confidence Sampling Strategy with dropout a active learning strategy where 
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
        \\mbox{argmin}_{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\sum_S(\\mbox{argmax}_j{(\\sigma(\\overrightarrow{\\boldsymbol{z}}))})}  
    
    where :math:`\\mathcal{U}` denotes the Data without lables i.e. `unlabeled_x` and :math:`k` is the `budget`.
    The drop out version uses the predict probability dropout function from the base strategy class to find the hypothesised labels.
    User can pass n_drop argument which denotes the number of times the probabilities will be calculated.
    The final probability is calculated by averaging probabilities obtained in all iteraitons. 
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled training dataset
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
        - **n_drop**: Number of dropout runs (int, optional)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(LeastConfidenceSamplingDropout, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob_dropout(unlabeled_buffer, self.n_drop)
        U = -probs.max(1)[0] # Max prob. negated => Largest score corresponds to smallest max prob (least confident prediction)
        return U