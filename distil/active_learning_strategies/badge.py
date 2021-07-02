from .strategy import Strategy
import numpy as np

import torch
from torch import nn
import random
import math

from scipy import stats


def init_centers(X, K, device):
    pdist = nn.PairwiseDistance(p=2)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            D2 = torch.flatten(D2)
            D2 = D2.cpu().numpy().astype(float)
        else:
            newD = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            newD = torch.flatten(newD)
            newD = newD.cpu().numpy().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    #gram = np.matmul(X[indsAll], X[indsAll].T)
    #val, _ = np.linalg.eig(gram)
    #val = np.abs(val)
    #vgt = val[val > 1e-2]
    return indsAll

class BADGE(Strategy):
    """
    This method is based on the paper Deep Batch Active Learning by Diverse, Uncertain Gradient 
    Lower Bounds :footcite:`DBLP-Badge`. According to the paper, this strategy, Batch Active 
    learning by Diverse Gradient Embeddings (BADGE), samples groups of points that are disparate 
    and high magnitude when represented in a hallucinated gradient space, a strategy designed to 
    incorporate both predictive uncertainty and sample diversity into every selected batch. 
    Crucially, BADGE trades off between uncertainty and diversity without requiring any hand-tuned 
    hyperparameters. Here at each round of selection, loss gradients are computed using the 
    hypothesised labels. Then to select the points to be labeled are selected by applying 
    k-means++ on these loss gradients. 
    
    Parameters
    ----------
    X: numpy array
        Present training/labeled data   
    Y: numpy array
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
        Specify optional parameters.
        `batch_size` 
        Batch size to be used inside strategy class (int, optional)
    """

    def __init__(self, X, Y, unlabeled_x, net, handler,nclasses, args):

        super(BADGE, self).__init__(X, Y, unlabeled_x, net, handler,nclasses, args)

    def select_per_batch(self, budget, batch_size):
        """
        Select points to label by using per-batch BADGE strategy

        Parameters
        ----------
        budget : int
            Number of indices to be selected from unlabeled set
        batch_size : int
            Size of batches to form

        Returns
        -------
        chosen: list
            List of selected data point indices with respect to unlabeled_x

        """
        
        # Compute gradient embeddings of each unlabeled point
        grad_embedding = self.get_grad_embedding(self.unlabeled_x,bias_grad=False)
        
        # Calculate number of batches to choose from, embedding dimension, and adjusted budget
        num_batches = math.ceil(grad_embedding.shape[0] / batch_size)
        embed_dim = grad_embedding.shape[1]
        batch_budget = math.ceil(budget / batch_size)
        
        # Instantiate list of lists of indices drawn from the possible range of the gradient embedding
        batch_indices_list = []
        draw_without_replacement = list(range(grad_embedding.shape[0]))
        
        while len(draw_without_replacement) > 0:
            
            if len(draw_without_replacement) < batch_size:
                batch_random_sample = draw_without_replacement
            else:
                batch_random_sample = random.sample(draw_without_replacement, batch_size)
        
            batch_indices_list.append(batch_random_sample)
            
            for index in batch_random_sample:
                draw_without_replacement.remove(index)
        
        # Instantiate batch average tensor
        gradBatchEmbedding = torch.zeros([num_batches, embed_dim]).to(self.device)
        
        # Calculate the average vector embedding of each batch
        for i in range(num_batches):
            
            indices = batch_indices_list[i]
            vec_avg = torch.zeros(embed_dim).to(self.device)
            for index in indices:
                vec_avg = vec_avg + grad_embedding[index]
            vec_avg = vec_avg / len(indices)
            
            gradBatchEmbedding[i] = vec_avg

        # Perform initial centers problem using new budget
        chosen_batch = init_centers(gradBatchEmbedding.cpu().numpy(), batch_budget, self.device)
        
        # For each chosen batch, construct the list of indices to return.
        chosen = []
        
        for batch_index in chosen_batch:
            
            indices_to_add = batch_indices_list[batch_index]
            chosen.extend(indices_to_add)

        return chosen

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

        gradEmbedding = self.get_grad_embedding(self.unlabeled_x,bias_grad=False)
        chosen = init_centers(gradEmbedding.cpu().numpy(), budget, self.device)
        return chosen