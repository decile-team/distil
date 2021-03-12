import copy

from torch.utils.data import DataLoader
from .strategy import Strategy
from ..utils import FixedWeightGradMatchStrategy
from ..utils import OMPGradMatchStrategy
from ..utils import SupervisedSelectHandler
    
# Define a GradMatch Active strategy
class GradMatchActive(Strategy):
    
    def __init__(self, X, Y, unlabeled_x, net, criterion, handler, nclasses, lrn_rate, selection_type, linear_layer, args={}):
        
        # Run super constructor
        super(GradMatchActive, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args)
        self.criterion = criterion
        self.lrn_rate = lrn_rate
        self.selection_type = selection_type
        self.linear_layer = linear_layer

    def select(self, budget, use_weights):
        
        # Compute hypothesize labels using model
        hypothesized_labels = self.predict(self.unlabeled_x)
        
        # Create a DataLoader from hypothesized labels and unlabeled points that will work with CORDS
        cords_handler = SupervisedSelectHandler(self.handler(self.unlabeled_x, hypothesized_labels.numpy(), False))        
        trainloader = DataLoader(cords_handler, shuffle=False, batch_size = self.args['batch_size'])

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