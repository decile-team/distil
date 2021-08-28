from .score_streaming_strategy import ScoreStreamingStrategy

class MarginSampling(ScoreStreamingStrategy):
    
    """
    Implements the Margin Sampling Strategy a active learning strategy similar to Least Confidence 
    Sampling Strategy. While least confidence only takes into consideration the maximum probability, 
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
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(MarginSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob(unlabeled_buffer)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:,1] - probs_sorted[:, 0] # Margin negated => Largest score corresponds to smallest margin
        return U
