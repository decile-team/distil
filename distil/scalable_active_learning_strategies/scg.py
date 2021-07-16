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
        gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        verbose = self.args['verbose'] if 'verbose' in self.args else False
        embedding_type = self.args['embedding_type'] if 'embedding_type' in self.args else "gradients"
        if(embedding_type=="features"):
            layer_name = self.args['layer_name'] if 'layer_name' in self.args else "avgpool"

        #Compute Embeddings
        if(embedding_type == "gradients"):
            unlabeled_data_embedding = self.get_grad_embedding(self.unlabeled_dataset, True, gradType)
            private_embedding = self.get_grad_embedding(self.private_dataset, False, gradType)
        elif(embedding_type == "features"):
            unlabeled_data_embedding = self.get_feature_embedding(self.unlabeled_dataset, True, layer_name)
            private_embedding = self.get_feature_embedding(self.private_dataset, False, layer_name)
        else:
            raise ValueError("Provided representation must be one of gradients or features")
        
        #Compute image-image kernel
        data_sijs = submodlib.helper.create_kernel(X=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")
        #Compute private-private kernel
        if(self.args['smi_function']=='logdetcg'):
            private_private_sijs = submodlib.helper.create_kernel(X=private_embedding.cpu().numpy(), metric=metric, method="sklearn")
        #Compute image-private kernel
        private_sijs = submodlib.helper.create_kernel(X=private_embedding.cpu().numpy(), X_rep=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")
        
        if(self.args['scg_function']=='flcg'):
            obj = submodlib.FacilityLocationConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_privates=private_embedding.shape[0],  
                                                                      data_sijs=data_sijs, 
                                                                      private_sijs=private_sijs, 
                                                                      privacyHardness=nu)
        
        if(self.args['scg_function']=='gccg'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.GraphCutConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_privates=private_embedding.shape[0],
                                                                      lambdaVal=lambdaVal,  
                                                                      data_sijs=data_sijs, 
                                                                      private_sijs=private_sijs, 
                                                                      privacyHardness=nu)
        if(self.args['scg_function']=='logdetcg'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.LogDeterminantConditionalGainFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_privates=private_embedding.shape[0],
                                                                      lambdaVal=lambdaVal,  
                                                                      data_sijs=data_sijs, 
                                                                      private_sijs=private_sijs,
                                                                      private_private_sijs=private_private_sijs, 
                                                                      privacyHardness=nu)

        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]
        return greedyIndices