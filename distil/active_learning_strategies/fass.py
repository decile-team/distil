from .strategy import Strategy
import copy
import datetime
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math
import random
from torch.distributions import Categorical
sys.path.append('../')
from distil.utils.submodular import SubmodularFunction

class FASS(Strategy):
    """
    Implementation of FASS strategy:footcite:`pmlr-v37-wei15` to select data points for active learning.
    This class extends :class:`active_learning_strategies.strategy.Strategy`.
    
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

    def select(self, budget):
        """
        Select next set of points

        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set

        Returns
        ----------
        return_indices: list
            List of selected data point indexes with respect to unlabeled_x
        """ 

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        submod_choices = ['facility_location', 'graph_cut', 'saturated_coverage', 'sum_redundancy', 'feature_based']
        if self.submod not in submod_choices:
            raise ValueError('Submodular function is invalid, Submodular functions can only be '+ str(submod_choices))
        selection_type = ['PerClass', 'Supervised', 'Full']
        if self.selection_type not in selection_type:
            raise ValueError('Selection type is invalid, Selection type can only be '+ str(selection_type))

        curr_X_trn = self.unlabeled_x
        cached_state_dict = copy.deepcopy(self.model.state_dict())
        predicted_y = self.predict(curr_X_trn)  # Hypothesised Labels
        soft = self.predict_prob(curr_X_trn)    #Probabilities of each class

        entropy2 = Categorical(probs = soft).entropy()
        
        if 5*budget < entropy2.shape[0]:
            values,indices = entropy2.topk(5*budget)
        else:
            indices = [i for i in range(entropy2.shape[0])]    
        # curr_X_trn = torch.from_numpy(curr_X_trn)
        curr_X_trn_embeddings = self.get_embedding(curr_X_trn)
        curr_X_trn_embeddings  = curr_X_trn_embeddings.reshape(curr_X_trn.shape[0], -1)

        submodular = SubmodularFunction(device, curr_X_trn_embeddings[indices], predicted_y[indices],\
            curr_X_trn.shape[0], 32, self.submod, self.selection_type)
        dsf_idxs_flag_val = submodular.lazy_greedy_max(budget, cached_state_dict)

        #Mapping to original indices
        return_indices = []
        for val in dsf_idxs_flag_val:
            append_val = val
            return_indices.append(indices[append_val])
        return return_indices