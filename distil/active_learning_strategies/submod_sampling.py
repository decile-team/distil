from .strategy import Strategy

import torch

from ..utils.submodular import SubmodularFunction
from ..utils.disparity_functions import DisparityFunction
from ..utils.similarity_mat import SimilarityComputation
from ..utils.dpp import dpp

class SubmodSampling(Strategy):
    """

    This strategy uses one of  the submodular functions viz. 'facility_location', 'graph_cut', 
    'saturated_coverage', 'sum_redundancy', 'feature_based' :footcite:`iyer2021submodular` 
    or Disparity-sum, Disparity-min :footcite:`dasgupta-etal-2013-summarization` or 
    DPP :footcite:`NEURIPS2018_dbbf603f` is used to select the points to be labeled. These 
    techniques can be applied directly to the features/embeddings or on the gradients of the 
    loss functions.

    
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
    handler: class object
        It should be a subclass of torch.utils.data.Dataset i.e, have __getitem__ and __len__ methods implemented, so that is could be passed to pytorch DataLoader.Could be instance of handlers defined in `distil.utils.DataHandler` or something similar.
    nclasses: int 
        No. of classes in tha dataset
    typeOf: str
        Choice of submodular function - 'facility_location' | 'graph_cut' | 'saturated_coverage' | 'sum_redundancy' | 'feature_based'\
            | 'Disparity-min' | 'Disparity-sum' | 'DPP'
    selection_type : str
       selection strategy - 'Full' |'PerClass' | 'Supervised' 
    if_grad : boolean, optional
        Determines if gradients to be used for subset selection. Default is False.
    args: dictionary
        This dictionary should have keys 'batch_size' and  'lr'. 
        'lr' should be the learning rate used for training. 'batch_size'  'batch_size' should be such 
        that one
    kernel_batch_size: int, optional
        For 'Diversity' and 'FacLoc' regualrizer versions, similarity kernel is to be computed, which 
        entails creating a 3d torch tensor of dimenssions kernel_batch_size*kernel_batch_size*
        feature dimenssion.Again kernel_batch_size should be such that one can exploit the benefits of 
        tensorization while honouring the resourse constraits.      
    """

    def __init__(self,X, Y,unlabeled_x, net, handler, nclasses,typeOf,selection_type,\
        if_grad=False,args={},kernel_batch_size = 200): # 
        super(SubmodSampling, self).__init__(X, Y, unlabeled_x, net, handler,nclasses, args)

        self.typeOf = typeOf
        self.if_grad = if_grad
        self.selection_type = selection_type
        self.kernel_batch_size = kernel_batch_size

    def _compute_per_element_grads(self):
        
        self.grads_per_elem = self.get_grad_embedding(self.unlabeled_x)
    
    def select(self, budget):

        """
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        
        Returns
        ----------
        chosen: list
            List of selected data point indexes with respect to unlabeled_x
        """ 

        if self.if_grad:
            self._compute_per_element_grads()
            selection_matrix = self.grads_per_elem
        else:
            selection_matrix = self.unlabeled_x

        submod_choices = ['facility_location', 'graph_cut', 'saturated_coverage', 'sum_redundancy',\
             'feature_based','Disparity-min', 'Disparity-sum','DPP']
        
        if self.typeOf not in submod_choices:
            raise ValueError('Submodular function is invalid, Submodular functions can only be '+ str(submod_choices))
        selection_type = ['PerClass', 'Supervised','Full']
        if self.selection_type not in selection_type:
            raise ValueError('Selection type is invalid, Selection type can only be '+ str(selection_type))

        predicted_y = self.predict(self.unlabeled_x)  # Hypothesised Labels
        
        if self.typeOf in submod_choices[:-3]:
            func = SubmodularFunction(self.device, selection_matrix, predicted_y,\
                len(predicted_y), self.kernel_batch_size, self.typeOf, self.selection_type)
            
            greedySet = func.lazy_greedy_max(budget)

        elif self.typeOf in submod_choices[-3:-1]:
            if self.typeOf in submod_choices[-3]:
                sub_type = "min"
            else:
                sub_type = "sum"

            func = DisparityFunction(self.device, selection_matrix, predicted_y, len(predicted_y),\
                 self.kernel_batch_size,sub_type, self.selection_type)

            greedySet = func.naive_greedy_max(budget)

        elif self.typeOf == submod_choices[-1]:
            simil = SimilarityComputation(self.device, selection_matrix, predicted_y, len(predicted_y),\
                 self.kernel_batch_size)

            classes, no_elements = torch.unique(predicted_y, return_counts=True)
            len_unique_elements = no_elements.shape[0]
            per_class_bud = int(budget / len(classes))
            final_per_class_bud = []
            _, sorted_indices = torch.sort(no_elements, descending = True)
            
            if self.selection_type == 'PerClass':
        
                total_idxs = 0
                for n_element in no_elements:
                    final_per_class_bud.append(min(per_class_bud, torch.IntTensor.item(n_element)))
                    total_idxs += min(per_class_bud, torch.IntTensor.item(n_element))
                
                if total_idxs < budget:
                    bud_difference = budget - total_idxs
                    for i in range(len_unique_elements):
                        available_idxs = torch.IntTensor.item(no_elements[sorted_indices[i]])-per_class_bud 
                        final_per_class_bud[sorted_indices[i]] += min(bud_difference, available_idxs)
                        total_idxs += min(bud_difference, available_idxs)
                        bud_difference = budget - total_idxs
                        if bud_difference == 0:
                            break

                greedySet = []
                for i in range(len_unique_elements):
                    idxs = torch.where(predicted_y == classes[i])[0]
                    simil.compute_score(idxs)

                    greedyList = dpp(simil.dist_mat.cpu().numpy(),final_per_class_bud[i])
                    greedySet.extend(idxs[greedyList])            
            
            elif self.selection_type == 'Full':

                greedySet = []
                
                simil.compute_score([i for i in range(len(predicted_y))])

                greedySet = dpp(simil.dist_mat.cpu().numpy(),budget)    

            elif self.selection_type == 'Supervised':
                 raise ValueError('Please use Full or PerClass')
             

        return greedySet