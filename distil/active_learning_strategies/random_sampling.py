import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):

    """
    Implementation of Random Sampling Strategy. This strategy is often used as a baseline, 
    where we pick a set of unlabeled points randomly.
    
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
        
        super(RandomSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
    def select(self, budget):

        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	        

        rand_idx = np.random.permutation(len(self.unlabeled_dataset))[:budget]
        rand_idx = rand_idx.tolist()
        return rand_idx