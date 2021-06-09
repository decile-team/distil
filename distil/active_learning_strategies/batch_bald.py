import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .strategy import Strategy
from ..utils.batch_bald.consistent_mc_dropout import ConsistentMCDropout
from ..utils.batch_bald.batchbald import get_batchbald_batch

class BatchBALDDropout(Strategy):
    """
    Implementation of BatchBALD Strategy.
    This class extends :class:`active_learning_strategies.strategy.Strategy`
    to include the MC sampling technique used to select data points for active learning.

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
        Number of dropout runs to use to generate MC samples (int, optional)
        
        n_samples
        Number of samples to use in computing joint entropy (int, optional)
    """
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        
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
        
        super(BatchBALDDropout, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)

    def do_MC_dropout_before_linear(self, X, n_drop):
        
        # Procure a loader on the supplied dataset
        loader_te = DataLoader(self.handler(X), shuffle=False, batch_size = self.args['batch_size'])
        
        # Check that there is a linear layer attribute in the supplied model
        try:
            getattr(self.model, self.mod_inject)
        except:
            raise ValueError(F"Model does not have attribute {self.mod_inject} as the last layer")
            
        # Make sure that the model is in eval mode
        self.model.eval()
            
        # Store the linear layer in a temporary variable
        lin_layer_temp = getattr(self.model, self.mod_inject)
        
        # Inject dropout into the model by using ConsistentMCDropout module from BatchBALD repo
        dropout_module = ConsistentMCDropout()
        dropout_injection = torch.nn.Sequential(dropout_module, lin_layer_temp)
        setattr(self.model, self.mod_inject, dropout_injection)

        # Create a tensor that will store the probabilities 
        probs = torch.zeros([n_drop, X.shape[0], self.target_classes]).to(self.device)
        with torch.no_grad():
            for i in range(n_drop):
                for x, idxs in loader_te:
                    x = x.to(self.device)
                    out = self.model(x)
                    probs[i][idxs] = F.softmax(out, dim=1)
        
        # Transpose the probs to match BatchBALD interface
        probs = probs.permute(1,0,2)
        
        # Restore the linear layer
        setattr(self.model, self.mod_inject, lin_layer_temp)
        
        # Return the MC samples for each data instance
        return probs

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
		
        # Get the MC samples from 
        probs = self.do_MC_dropout_before_linear(self.unlabeled_x, self.n_drop)
        
        # Compute the log probabilities to match BatchBALD interface
        log_probs = torch.log(probs)
        
        # Use BatchBALD interface to select the new points. 
        candidate_batchbald_batch = get_batchbald_batch(log_probs, budget, self.n_samples, device=self.device)        
        
        # Return the selected indices
        return candidate_batchbald_batch.indices