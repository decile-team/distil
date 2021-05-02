import copy

from torch.utils.data import DataLoader
from .strategy import Strategy
from ..utils.supervised_strategy_wrappers import CRAIGStrategy
from ..utils.supervised_strategy_wrappers import SupervisedSelectHandler

# Define a GradMatch Active strategy
class CRAIGActive(Strategy):
    
    """
    This is an implementation of an active learning variant of CRAIG from the paper Coresets for Data-efficient 
    Training of Machine Learning Models :footcite:`Mirzasoleiman2020craig`. This algorithm calculates hypothesized 
    labels for each of the unlabeled points and feeds this hypothesized set to the original CRAIG algorithm. The 
    selected points from CRAIG are used as the queried points for this algorithm.

    
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
        The learning rate used in training. Used by the CRAIG algorithm.
    selection_type: string
        Should be one of "PerClass", "Supervised", or "PerBatch". Selects which approximation method is used.
    linear_layer: bool
        Sets whether to include the last linear layer parameters as part of the gradient computation.
    args: dictionary
        This dictionary should have keys 'batch_size' and  'lr'. 
        'lr' should be the learning rate used for training. 'batch_size'  'batch_size' should be such 
        that one can exploit the benefits of tensorization while honouring the resourse constraits.
    """
    
    def __init__(self, X, Y, unlabeled_x, net, criterion, handler, nclasses, lrn_rate, selection_type, linear_layer, args={}):
        
        # Run super constructor
        super(CRAIGActive, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)
        self.criterion = criterion
        self.lrn_rate = lrn_rate
        self.selection_type = selection_type
        self.linear_layer = linear_layer

    def select(self, budget):
        
        """
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        
        Returns
        ----------
        subset_idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """ 
        
        # Compute hypothesize labels using model
        hypothesized_labels = self.predict(self.unlabeled_x)
        
        # Create a DataLoader from hypothesized labels and unlabeled points that will work with CORDS
        cords_handler = SupervisedSelectHandler(self.handler(self.unlabeled_x, hypothesized_labels.numpy(), False))        
        trainloader = DataLoader(cords_handler, shuffle=False, batch_size = self.args['batch_size'])

        # Match on the hypothesized labels
        validloader = trainloader
        
        # Perform selection
        cached_state_dict = copy.deepcopy(self.model.state_dict())
        clone_dict = copy.deepcopy(self.model.state_dict())

        # Create CORDS CRAIGStrategy to select datapoints
        setf_model = CRAIGStrategy(trainloader, validloader, self.model, self.criterion, self.device, self.target_classes, self.linear_layer, False, self.selection_type)
        subset_idxs, _ = setf_model.select(budget, clone_dict)
        self.model.load_state_dict(cached_state_dict)
        return subset_idxs