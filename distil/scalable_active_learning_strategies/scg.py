from .strategy import Strategy
import numpy as np

import torch
from torch import nn
from scipy import stats
import submodlib

class SCG(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, private_dataset, net, nclasses, args={}): #
        
        super(SCG, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        
        self.private_dataset = private_dataset

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

        #Get hyperparameters from args dict
        optimizer = self.args['optimizer'] if 'optimizer' in self.args else 'NaiveGreedy'
        metric = self.args['metric'] if 'metric' in self.args else 'cosine'
        nu = self.args['nu'] if 'nu' in self.args else 1
        gradType = self.args['gradType'] if 'gradType' in self.args else "linear"
        stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        verbose = self.args['verbose'] if 'verbose' in self.args else False

        #Compute Embeddings
        unlabeled_data_embedding = self.get_grad_embedding(self.unlabeled_dataset, True, gradType)
        private_embedding = self.get_grad_embedding(self.private_dataset, True, gradType)
        
        if(self.args['scg_function']=='flcg'):
            obj = submodlib.FacilityLocationConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_privates=private_embedding.shape[0],  
                                                                      data=unlabeled_data_embedding, 
                                                                      privateData=private_embedding, 
                                                                      metric=metric,
                                                                      privacyHardness=nu)
        
        if(self.args['scg_function']=='gccg'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.GraphCutConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_privates=private_embedding.shape[0],
                                                                      lambdaVal=lambdaVal,  
                                                                      data=unlabeled_data_embedding, 
                                                                      privateData=private_embedding, 
                                                                      metric=metric,
                                                                      privacyHardness=nu)
        if(self.args['scg_function']=='logdetcg'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.LogDeterminantConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_privates=private_embedding.shape[0],
                                                                      lambdaVal=lambdaVal,  
                                                                      data=unlabeled_data_embedding, 
                                                                      privateData=private_embedding, 
                                                                      metric=metric,
                                                                      privacyHardness=nu)

        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]
        return greedyIndices