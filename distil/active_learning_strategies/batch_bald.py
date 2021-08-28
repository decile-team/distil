import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .strategy import Strategy
from ..utils.batch_bald.consistent_mc_dropout import ConsistentMCDropout
from ..utils.batch_bald.batchbald import get_batchbald_batch

class BatchBALDDropout(Strategy):
    """
    Implementation of BatchBALD Strategy :footcite:`kirsch2019batchbald`, which refines 
    the original BALD acquisition to the batch setting using a new acquisition function.
    This class extends :class:`active_learning_strategies.strategy.Strategy`
    to include a MC sampling technique based on the sampling techniques used in their paper.
    
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
        - **n_samples**: Number of samples to use in computing joint entropy (int, optional)
    """
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        """
        Constructor method
        """
        
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 40
            
        if 'n_samples' in args:
            self.n_samples = args['n_samples']
        else:
            self.n_samples = 1000
        
        if 'mod_inject' in args:
            self.mod_inject = args['mod_inject']
        else:
            self.mod_inject = 'linear'
        
        super(BatchBALDDropout, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args={})

    def do_MC_dropout_before_linear(self, unlabeled_dataset, n_drop):
        
        # Procure a loader on the supplied dataset
        loader_te = DataLoader(unlabeled_dataset, shuffle=False, batch_size = self.args['batch_size'])
        
        # Check that there is a linear layer attribute in the supplied model
        try:
            getattr(self.model, self.mod_inject)
        except:
            raise ValueError(F"Model does not have attribute {self.mod_inject} as the last layer")
            
        # Store the linear layer in a temporary variable
        lin_layer_temp = getattr(self.model, self.mod_inject)
        
        # Inject dropout into the model by using ConsistentMCDropout module from BatchBALD repo
        dropout_module = ConsistentMCDropout()
        dropout_injection = torch.nn.Sequential(dropout_module, lin_layer_temp)
        setattr(self.model, self.mod_inject, dropout_injection)

        # Make sure that the model is in eval mode
        self.model.eval()
        
        # For safety, explicitly set the dropout module to be in evaluation mode
        dropout_module.train(mode=False)
        
        # Create a tensor that will store the probabilities 
        probs = torch.zeros([n_drop, len(unlabeled_dataset), self.target_classes]).to(self.device)
        with torch.no_grad():
            for i in range(n_drop):
                evaluated_points = 0
                
                # In original BatchBALD code, inference samples were predicted in a single forward pass via an additional forward parameter.
                # Hence, only 1 mask needed to be generated during eval time for consistent MC sampling (as there was only 1 pass). Here, 
                # our models do not assume this forward parameter. Hence, we must have a different generated mask for each PASS of the 
                # dataset. Note, however, that the mask is CONSISTENT within a pass, which is functionally equivalent to the original 
                # BatchBALD code.
                dropout_module.reset_mask()
                
                for x in loader_te:
                    idxs = [iter_index for iter_index in range(evaluated_points, evaluated_points + len(x))]
                    x = x.to(self.device)
                    out = self.model(x)
                    probs[i][idxs] = F.softmax(out, dim=1)
                    evaluated_points += len(x)
        
        # Transpose the probs to match BatchBALD interface
        probs = probs.permute(1,0,2)
        
        # Restore the linear layer
        setattr(self.model, self.mod_inject, lin_layer_temp)
        
        # Return the MC samples for each data instance
        return probs

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
		
        # Move the model to the correct device
        self.model = self.model.to(self.device)
        
        # Get the MC samples from 
        probs = self.do_MC_dropout_before_linear(self.unlabeled_dataset, self.n_drop)
        
        # Compute the log probabilities to match BatchBALD interface
        log_probs = torch.log(probs)
        
        # Use BatchBALD interface to select the new points. 
        candidate_batchbald_batch = get_batchbald_batch(log_probs, budget, self.n_samples, device=self.device)        
        
        # Return the selected indices
        return candidate_batchbald_batch.indices