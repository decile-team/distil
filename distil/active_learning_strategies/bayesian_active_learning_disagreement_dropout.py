import torch

from .score_streaming_strategy import ScoreStreamingStrategy

class BALDDropout(ScoreStreamingStrategy):
    
    """
    Implements Bayesian Active Learning by Disagreement (BALD) Strategy :footcite:`houlsby2011bayesian`,
    which assumes a Basiyan setting and selects points which maximise the mutual information 
    between the predicted labels and model parameters. This implementation is an adaptation for a 
    non-bayesian setting, with the assumption that there is a dropout layer in the model.
    
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
        - **n_drop**: Number of dropout runs to use to generate MC samples (int, optional)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(BALDDropout, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob_dropout_split(unlabeled_buffer, self.n_drop)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        
        # Previous strategy sorts and takes smallest (entropy2 - entropy1). 
        # This one will take largest (entropy1 - entropy2) => smallest (entropy2-entropy1)
        U = entropy1 - entropy2 
        return U