import numpy as np
import torch

from torch.utils.data import SequentialSampler, BatchSampler

class SimilarityComputation():

    """
    Implementation of Submodular Function.
    This class allows you to use different submodular functions
            
    Parameters
    ----------
    device: str
        Device to be used, cpu|gpu
    x_trn: torch tensor
        Data on which submodular optimization should be applied
    y_trn: torch tensor
        Labels of the data 
    model: class
        Model architecture used for training
    N_trn: int
        Number of samples in dataset
    batch_size: int
        Batch size to be used for optimization
    if_convex: bool
        If convex or not
    submod: str
        Choice of submodular function - 'facility_location' | 'graph_cut' | 'saturated_coverage' | 'sum_redundancy' | 'feature_based'
    selection_type: str
        Type of selection - 'PerClass' | 'Supervised' | 'Full'
    """

    def __init__(self, device, x_trn, y_trn, N_trn, batch_size):
        self.x_trn = x_trn
        self.y_trn = y_trn
        self.device = device
        self.N_trn = N_trn
        self.batch_size = batch_size
        
    def distance(self, x, y, exp=2):
        """
        Compute the distance.
 
        Parameters
        ----------
        x: Tensor
            First input tensor
        y: Tensor
            Second input tensor
        exp: float, optional
            The exponent value (default: 2)
            
        Returns
        ----------
        dist: Tensor
            Output tensor 
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def get_index(self, data, data_sub):
        """
        Returns indexes of the rows.
 
        Parameters
        ----------
        data: numpy array
            Array to find indexes from
        data_sub: numpy array
            Array of data points to find indexes for
            
        Returns
        ----------
        greedyList: list
            List of indexes 
        """

        greedyList = []
        for row in data_sub:
            idx_map = np.where(np.all(row == data, axis=1))[0]
            for val in idx_map:
                if val not in greedyList:
                    greedyList.append(val)
                    break

        return greedyList


    def compute_score(self, idxs):

        """
        Compute the score of the indices.
        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        idxs: list
            The indices
        """

        self.N = 0
        g_is = []
        x_temp = self.x_trn[idxs]
        y_temp = self.y_trn[idxs]
        batch_wise_indices = np.array(
            [list(BatchSampler(SequentialSampler(np.arange(len(y_temp))), self.batch_size, drop_last=False))][0])
        with torch.no_grad():
            for batch_idx in batch_wise_indices:
                inputs_i = x_temp[batch_idx].type(torch.float)
                target_i = y_temp[batch_idx]
                inputs_i, target_i = inputs_i.to(self.device), target_i.to(self.device)
                self.N += inputs_i.size()[0]
                g_is.append(inputs_i)
                
            self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j).cpu()
        #self.dist_mat = self.dist_mat.cpu().numpy()

    