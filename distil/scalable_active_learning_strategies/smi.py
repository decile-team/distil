from distil.scalable_active_learning_strategies.strategy import Strategy
import numpy as np

import torch
from torch import nn
from scipy import stats
import submodlib

class SMI(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, query_dataset, net, nclasses, args={}): #
        
        super(SMI, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        
        self.query_dataset = query_dataset

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
        gradType = self.args['gradType'] if 'gradType' in self.args else "linear"
        stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        verbose = self.args['verbose'] if 'verbose' in self.args else False
        

        #Compute Embeddings
        unlabeled_data_embedding = self.get_grad_embedding(self.unlabeled_dataset, True, gradType)
        query_embedding = self.get_grad_embedding(self.query_dataset, True, gradType)


        
        if(self.args['smi_function']=='fl1mi'):
            obj = submodlib.FacilityLocationMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=unlabeled_data_embedding, 
                                                                      queryData=query_embedding, 
                                                                      metric=metric, 
                                                                      magnificationLambda=eta)

        if(self.args['smi_function']=='fl2mi'):
            obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=unlabeled_data_embedding, 
                                                                      queryData=query_embedding, 
                                                                      metric=metric, 
                                                                      magnificationLambda=eta)
        
        if(self.args['smi_function']=='com'):
            from submodlib_cpp import ConcaveOverModular
            obj = submodlib.ConcaveOverModularFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=unlabeled_data_embedding, 
                                                                      queryData=query_embedding, 
                                                                      metric=metric, 
                                                                      magnificationLambda=eta,
                                                                      mode=ConcaveOverModular.logarithmic)
        if(self.args['smi_function']=='gcmi'):
            obj = submodlib.GraphCutMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=unlabeled_data_embedding, 
                                                                      queryData=query_embedding, 
                                                                      metric=metric)
        if(self.args['smi_function']=='logdetmi'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.LogDeterminantMutualInformationFunction(n=unlabeled_data_embedding.shape[0],
                                                                    num_queries=query_embedding.shape[0], 
                                                                    data=unlabeled_data_embedding, 
                                                                    queryData=query_embedding, 
                                                                    metric=metric, 
                                                                    magnificationLambda=eta,
                                                                    lambdaVal=lambdaVal)

        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]
        return greedyIndices