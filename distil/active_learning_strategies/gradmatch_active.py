import copy

from torch.utils.data import DataLoader
from .strategy import Strategy
from ..utils.supervised_strategy_wrappers import FixedWeightGradMatchStrategy
from ..utils.supervised_strategy_wrappers import OMPGradMatchStrategy
from ..utils.supervised_strategy_wrappers import SupervisedSelectHandler
    
# Define a GradMatch Active strategy
class GradMatchActive(Strategy):
    
    """
    This is an implementation of an active learning variant of GradMatch from the paper GRAD-MATCH: A 
    Gradient Matching Based Data Subset Selection for Efficient Learning :footcite:`Killamsetty2021gradmatch`.
    This algorithm solves a fixed-weight version of the error term present in the paper by a greedy selection 
    algorithm akin to the original GradMatch's Orthogonal Matching Pursuit. The gradients computed are on the 
    hypothesized labels of the loss function and are matched to either the full gradient of these hypothesized 
    examples or a supplied validation gradient. The indices returned are the ones selected by this algorithm.

    .. math::
        Err(X_t, L, L_T, \\theta_t) = \\left |\\left| \\sum_{i \\in X_t} \\nabla_\\theta L_T^i (\\theta_t) - \\frac{k}{N} \\nabla_\\theta L(\\theta_t) \\right | \\right|

    where,

        - Each gradient is computed with respect to the last layer's parameters
        - :math:`\\theta_t` are the model parameters at selection round :math:`t`
        - :math:`X_t` is the queried set of points to label at selection round :math:`t`
        - :math:`k` is the budget
        - :math:`N` is the number of points contributing to the full gradient :math:`\\nabla_\\theta L(\\theta_t)`
        - :math:`\\nabla_\\theta L(\\theta_t)` is either the complete hypothesized gradient or a validation gradient
        - :math:`\\sum_{i \\in X_t} \\nabla_\\theta L_T^i (\\theta_t)` is the subset's hypothesized gradient with :math:`|X_t| = k`


    Parameters
    ----------
    X: Numpy array 
        Features of the labled set of points 
    Y: Numpy array
        Lables of the labled set of points 
    unlabeled_x: Numpy array
        Features of the unlabled set of points 
    net: class object
        Model architecture used for training. Could be instance of models defined in `distil.utils.models` or something similar.
    criterion: class object
        The loss type used in training. Could be instance of torch.nn.* losses or torch functionals.
    handler: class object
        It should be a subclass of torch.utils.data.Dataset i.e, have __getitem__ and __len__ methods implemented, so that is could be passed to pytorch DataLoader.Could be instance of handlers defined in `distil.utils.DataHandler` or something similar.
    nclasses: int 
        No. of classes in tha dataset
    lrn_rate: float
        The learning rate used in training. Used by the original GradMatch algorithm.
    selection_type: string
        Should be one of "PerClass" or "PerBatch". Selects which approximation method is used.
    linear_layer: bool
        Sets whether to include the last linear layer parameters as part of the gradient computation.
    args: dictionary
        This dictionary should have keys 'batch_size' and  'lr'. 
        'lr' should be the learning rate used for training. 'batch_size'  'batch_size' should be such 
        that one can exploit the benefits of tensorization while honouring the resourse constraits.
    valid: boolean
        Whether validation set is passed or not
    X_val: Numpy array, optional
        Features of the points in the validation set. Mandatory if `valid=True`.
    Y_val:Numpy array, optional
        Lables of the points in the validation set. Mandatory if `valid=True`.
    """
    
    def __init__(self, X, Y, unlabeled_x, net, criterion, handler, nclasses, lrn_rate, selection_type, linear_layer, args={}, valid=False, X_val=None, Y_val=None):
        
        # Run super constructor
        super(GradMatchActive, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)
        self.criterion = criterion
        self.lrn_rate = lrn_rate
        self.selection_type = selection_type
        self.linear_layer = linear_layer
        self.valid = valid
        if valid:
            self.X_Val = X_val
            self.Y_Val = Y_val

    def select(self, budget, use_weights):
        
        """
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        use_weights: bool
            Whether to use fixed-weight version (false) or OMP version (true)
        
        Returns
        ----------
        subset_idxs: list
            List of selected data point indexes with respect to unlabeled_x and, if use_weights is true, the weights associated with each point
        """ 
        
        # Compute hypothesize labels using model
        hypothesized_labels = self.predict(self.unlabeled_x)
        
        # Create a DataLoader from hypothesized labels and unlabeled points that will work with CORDS
        cords_handler = SupervisedSelectHandler(self.handler(self.unlabeled_x, hypothesized_labels.numpy(), False))        
        trainloader = DataLoader(cords_handler, shuffle=False, batch_size = self.args['batch_size'])
        if(self.valid):
            cords_val_handler = SupervisedSelectHandler(self.handler(self.X_Val, self.Y_Val, False))
            validloader = DataLoader(cords_val_handler, shuffle=False, batch_size = self.args['batch_size'])
        else:
            validloader = trainloader
        # Perform selection
        cached_state_dict = copy.deepcopy(self.model.state_dict())
        clone_dict = copy.deepcopy(self.model.state_dict())

        if use_weights:
            # Create OMPGradMatchStrategy to select datapoints
            setf_model = OMPGradMatchStrategy(trainloader, validloader, self.model, self.criterion, self.lrn_rate, self.device, self.target_classes, self.linear_layer, self.selection_type, True, lam=1)
            subset_idxs, gammas = setf_model.select(budget, clone_dict)
            self.model.load_state_dict(cached_state_dict)
            return subset_idxs, gammas
        else:
            # Create FixedWeightGradMatchStrategy to select datapoints
            setf_model = FixedWeightGradMatchStrategy(trainloader, validloader, self.model, self.criterion, self.lrn_rate, self.device, self.target_classes, self.linear_layer, self.selection_type, True)
            subset_idxs = setf_model.select(budget, clone_dict)
            self.model.load_state_dict(cached_state_dict)
            return subset_idxs