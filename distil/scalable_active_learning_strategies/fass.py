from .entropy_sampling import EntropySampling
from .least_confidence import LeastConfidence
from .margin_sampling import MarginSampling
from .submod_sampling import SubmodularSampling
from .strategy import Strategy

from torch.utils.data import Subset

class FASS(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(FASS, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'submod_args' in args:
            self.submod_args = args['submod_args']
        else:
            self.submod_args = {'submod': 'facility_location',
                                'metric': 'cosine'}
            self.args['submod_args'] = self.submod_args
        
        if 'uncertainty_measure' in args:
            self.uncertainty_measure = args['uncertainty_measure']
        else:
            self.uncertainty_measure = 'entropy'
        
    def select(self, budget, top_n=5):
        
        # Determine if top_n * budget points can be drawn; otherwise, set filtered set size 
        # to be the size of the unlabeled_dataset
        filtered_set_size = min(budget * top_n, len(self.unlabeled_dataset))

        # Now, select the top 'filtered_set_size' most uncertain points using the 
        # specified measure of uncertainty (already implemented in strategies!)
        if self.uncertainty_measure == 'entropy':
            uncertainty_strategy = EntropySampling(self.labeled_dataset, self.unlabeled_dataset, self.model, self.target_classes, self.args)
        elif self.uncertainty_measure == 'least_confidence':
            uncertainty_strategy = LeastConfidence(self.labeled_dataset, self.unlabeled_dataset, self.model, self.target_classes, self.args)
        elif self.uncertainty_measure == 'margin':
            uncertainty_strategy = MarginSampling(self.labeled_dataset, self.unlabeled_dataset, self.model, self.target_classes, self.args)
        else:
            raise ValueError("uncertainty_measure must be one of 'entropy', 'least_confidence', or 'margin'")
        
        filtered_idxs = uncertainty_strategy.select(filtered_set_size)
        
        # Now, use submodular selection to choose points from the filtered subset.
        # Ensure the representation type is in the submod_args dict.
        if 'representation' not in self.submod_args:
            self.submod_args['representation'] = 'linear'
            
        filtered_unlabeled_set = Subset(self.unlabeled_dataset, filtered_idxs)
        submodular_selection_strategy = SubmodularSampling(self.labeled_dataset, filtered_unlabeled_set, self.model, self.target_classes, self.args)
        greedy_indices = submodular_selection_strategy.select(budget)
        
        # Lastly, map the indices of the filtered set to the indices of the unlabeled set
        selected_indices = [filtered_idxs[x] for x in greedy_indices]
        
        return selected_indices        