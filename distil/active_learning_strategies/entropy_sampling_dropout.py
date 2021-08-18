import torch

from .score_streaming_strategy import ScoreStreamingStrategy

class EntropySamplingDropout(ScoreStreamingStrategy):
    
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
        ENTROPY = -\\sum_j \\sigma(z_j)*\\log(\\sigma(z_j))
        
    The algorithm then selects `budget` no. of elements with highest **ENTROPY**.
    
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
        
        super(EntropySamplingDropout, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob_dropout(unlabeled_buffer, self.n_drop)
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(1)
        return U