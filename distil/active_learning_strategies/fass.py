from .strategy import Strategy
from torch.distributions import Categorical
from ..utils.submodular import SubmodularFunction

class FASS(Strategy):
    """
    Implements FASS :footcite:`pmlr-v37-wei15` combines the uncertainty sampling 
    method with a submodular data subset selection framework to label a subset of data points to 
    train a classifier. Here the based on the ‘top_n’ parameter, ‘top_n*budget’ most uncertain 
    parameters are filtered. On these filtered points one of  the submodular functions viz. 
    'facility_location' , 'graph_cut', 'saturated_coverage', 'sum_redundancy', 'feature_based' 
    is applied to get the final set of points.

    We select a subset :math:`F` of size :math:`\\beta` based on uncertainty sampling, such 
    that :math:`\\beta \\ge k`.
      
    Then select a subset :math:`S` by solving 
    
    .. math::
        \\max \\{f(S) \\text{ such that } |S| \\leq k, S \\subseteq F\\} 
    
    where :math:`k` is the is the `budget` and :math:`f` can be one of these functions - 
    'facility location' , 'graph cut', 'saturated coverage', 'sum redundancy', 'feature based'. 

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
        Specify optional parameters - `batch_size` 
        Batch size to be used inside strategy class (int, optional)

    submod: str
    Choice of submodular function - 'facility_location' | 'graph_cut' | 'saturated_coverage' | 'sum_redundancy' | 'feature_based'
    
    selection_type: str
    Choice of selection strategy - 'PerClass' | 'Supervised'
    """

    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):

        """
        Constructor method
        """
        
        if 'submod' in args:
            self.submod = args['submod']
        else:
            self.submod = 'facility_location'

        if 'selection_type' in args:
            self.selection_type = args['selection_type']
        else:
            self.selection_type = 'PerClass'
        super(FASS, self).__init__(X, Y, unlabeled_x, net, handler,nclasses, args)

    def select(self, budget,top_n=5):
        """
        Select next set of points

        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        top_n: float
            It is the multiper to the budget which decides the size of the data points on which \
            submodular functions will be applied. For example top_n = 5, if 5*budget points will
            be passed to the submodular functions.  
        Returns
        ----------
        return_indices: list
            List of selected data point indexes with respect to unlabeled_x
        """ 

        submod_choices = ['facility_location', 'graph_cut', 'saturated_coverage', 'sum_redundancy', 'feature_based']
        if self.submod not in submod_choices:
            raise ValueError('Submodular function is invalid, Submodular functions can only be '+ str(submod_choices))
        selection_type = ['PerClass', 'Supervised', 'Full']
        if self.selection_type not in selection_type:
            raise ValueError('Selection type is invalid, Selection type can only be '+ str(selection_type))

        if top_n < 1:
            raise ValueError('top_n parameter should be atleast 1' )


        curr_X_trn = self.unlabeled_x
        predicted_y = self.predict(curr_X_trn)  # Hypothesised Labels
        soft = self.predict_prob(curr_X_trn)    #Probabilities of each class

        entropy2 = Categorical(probs = soft).entropy()

        curr_size = int(top_n*budget)
        
        if curr_size < entropy2.shape[0]:
            values,indices = entropy2.topk(curr_size)
        else:
            indices = [i for i in range(entropy2.shape[0])]    
        # curr_X_trn = torch.from_numpy(curr_X_trn)
        curr_X_trn_embeddings = self.get_embedding(curr_X_trn)
        curr_X_trn_embeddings  = curr_X_trn_embeddings.reshape(curr_X_trn.shape[0], -1)

        submodular = SubmodularFunction(self.device, curr_X_trn_embeddings[indices], predicted_y[indices],\
            curr_X_trn.shape[0], 32, self.submod, self.selection_type)
        dsf_idxs_flag_val = submodular.lazy_greedy_max(budget)

        #Mapping to original indices
        return_indices = []
        for val in dsf_idxs_flag_val:
            append_val = val
            return_indices.append(indices[append_val])
        return return_indices