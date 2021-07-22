from .strategy import Strategy
import numpy as np

import torch
from torch import nn
from scipy import stats
import submodlib

class SCMI(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, query_dataset, private_dataset, net, nclasses, args={}): #
        
        super(SCMI, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        
        self.query_dataset = query_dataset
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
        eta = self.args['eta'] if 'eta' in self.args else 1
        nu = self.args['nu'] if 'nu' in self.args else 1
        gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        verbose = self.args['verbose'] if 'verbose' in self.args else False
        unlabeled = self.args['unlabeled'] if 'unlabeled' in self.args else False
        embedding_type = self.args['embedding_type'] if 'embedding_type' in self.args else "gradients"
        if(embedding_type=="features"):
            layer_name = self.args['layer_name'] if 'layer_name' in self.args else "avgpool"

        #Compute Embeddings
        if(embedding_type == "gradients"):
            unlabeled_data_embedding = self.get_grad_embedding(self.unlabeled_dataset, True, gradType)
            query_embedding = self.get_grad_embedding(self.query_dataset, False, gradType)
            private_embedding = self.get_grad_embedding(self.private_dataset, False, gradType)
        elif(embedding_type == "features"):
            unlabeled_data_embedding = self.get_feature_embedding(self.unlabeled_dataset, True, layer_name)
            query_embedding = self.get_feature_embedding(self.query_dataset, False, layer_name)
            private_embedding = self.get_feature_embedding(self.private_dataset, False, layer_name)
        else:
            raise ValueError("Provided representation must be one of gradients or features")

        #Compute image-image kernel
        data_sijs = submodlib.helper.create_kernel(X=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")
        #Compute query-query kernel
        if(self.args['smi_function']=='logdetmi'):
            query_query_sijs = submodlib.helper.create_kernel(X=query_embedding.cpu().numpy(), metric=metric, method="sklearn")
            private_private_sijs = submodlib.helper.create_kernel(X=private_embedding.cpu().numpy(), metric=metric, method="sklearn")
            query_private_sijs = submodlib.helper.create_kernel(X=private_embedding.cpu().numpy(), X_rep=query_embedding.cpu().numpy(), metric=metric, method="sklearn")
        #Compute image-query kernel
        query_sijs = submodlib.helper.create_kernel(X=query_embedding.cpu().numpy(), X_rep=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")
        private_sijs = submodlib.helper.create_kernel(X=private_embedding.cpu().numpy(), X_rep=unlabeled_data_embedding.cpu().numpy(), metric=metric, method="sklearn")
        
        if(self.args['scmi_function']=='flcmi'):
            obj = submodlib.FacilityLocationConditionalMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0],
                                                                      num_privates=private_embedding.shape[0], 
                                                                      data_sijs=data_sijs, 
                                                                      query_sijs=query_sijs, 
                                                                      private_sijs=private_sijs, 
                                                                      magnificationEta=eta,
                                                                      privacyHardness=nu)
    
        if(self.args['scmi_function']=='logdetcmi'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.LogDeterminantConditionalMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0],
                                                                      num_privates=private_embedding.shape[0], 
                                                                      data_sijs=data_sijs, 
                                                                      query_sijs=query_sijs, 
                                                                      private_sijs=private_sijs,
                                                                      query_query_sijs=query_query_sijs,
                                                                      private_private_sijs=private_private_sijs,
                                                                      query_private_sijs=query_private_sijs, 
                                                                      magnificationEta=eta,
                                                                      privacyHardness=nu,
                                                                      lambdaVal=lambdaVal)

        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]
        return greedyIndices