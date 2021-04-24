import numpy as np
import torch

from scipy.sparse import csr_matrix

from .similarity_mat import SimilarityComputation

class DisparityFunction(SimilarityComputation):

    """
    Implementation of Diparity Functions.
    This class allows you to use different Diparity functions
            
    Parameters
    ----------
    device: str
        Device to be used, cpu|gpu
    x_trn: torch tensor
        Data of which subset selected should be applied
    y_trn: torch tensor
        Labels of the data 
    N_trn: int
        Number of samples in dataset
    batch_size: int
        Batch size to be used for optimization
    if_convex: bool
        If convex or not
    dis_type: str
        Choice of Diparity function - 'sum' | 'min' 
    selection_type: str
        Type of selection - 'Full' |'PerClass' | 'Supervised' 
    """

    def __init__(self, device, x_trn, y_trn, N_trn, batch_size, dis_type, selection_type):
       
        super(DisparityFunction, self).__init__(device, x_trn, y_trn, N_trn, batch_size)
        
        self.dis_type = dis_type
        self.selection_type = selection_type

    def naive_greedy_max(self, budget):

        """
        Data selection method using different submodular optimization
        functions.
 
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        
        Returns
        ----------
        total_greedy_list: list
            List containing indices of the best datapoints 
        """

        classes, no_elements = torch.unique(self.y_trn, return_counts=True)
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

            total_greedy_list = []
            for i in range(len_unique_elements):
                idxs = torch.where(self.y_trn == classes[i])[0]

                numSelected = 1
                greedySet = [0]  
                remset = [n for n in range(1,final_per_class_bud[i])]

                self.compute_score(idxs)

                while(numSelected < final_per_class_bud[i]): 

                    if self.dis_type == "sum":

                        gains = self.dist_mat[remset][:, greedySet].sum(1)
                    elif self.dis_type == "min":

                        gains, _ = torch.max(self.dist_mat[remset][:, greedySet],dim=1)

                    temp_id = torch.argmin(gains).item()
                    best_id = remset[temp_id]
                    greedySet.append(best_id) 
                    remset.remove(best_id)  

                    numSelected +=1           
                
                total_greedy_list.extend(idxs[greedySet])

        elif self.selection_type == 'Full':

            self.compute_score([i for i in range(self.N_trn)])

            numSelected = 1
            greedySet = [0]  
            remset = [n for n in range(1,self.N_trn)]

            while(numSelected < budget): 

                if self.dis_type == "sum":

                    gains = self.dist_mat[remset][:, greedySet].sum(1)
                elif self.dis_type == "min":

                    gains, _ = torch.max(self.dist_mat[remset][:, greedySet],dim=1)

                temp_id = torch.argmin(gains).item()
                best_id = remset[temp_id]
                greedySet.append(best_id) 
                remset.remove(best_id)  

                numSelected +=1           
            
            total_greedy_list = greedySet

        elif self.selection_type == 'Supervised':
            
            for i in range(len(classes)):
                
                if i == 0:
                    idxs = torch.where(self.y_trn == classes[i])[0]
                    N = len(idxs)
                    self.compute_score(idxs)
                    row = idxs.repeat_interleave(N)
                    col = idxs.repeat(N)
                    data = self.dist_mat.flatten()
                else:
                    idxs = torch.where(self.y_trn == classes[i])[0]
                    N = len(idxs)
                    self.compute_score(idxs)
                    row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
                    col = torch.cat((col, idxs.repeat(N)), dim=0)
                    data = np.concatenate([data, self.dist_mat.flatten()], axis=0)
                
                
            sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
            
            total_greedy_list = []

            numSelected = 1
            greedySet = [0]  
            remset = [n for n in range(1,self.N_trn)]

            current_values = np.zeros(self.N_trn)

            #print(sparse_simmat.indptr)

            #current_values[sparse_simmat.indices[:sparse_simmat.indptr[1]]] =\ 
            #     sparse_simmat.data[:sparse_simmat.indptr[1]].todense()  

            current_values = sparse_simmat[0,:].todense()
            current_values = np.ravel(current_values.sum(axis=0))

            current_values[0] = np.inf          

            while(numSelected < budget): 

                best_id = np.argmin(current_values).item()
                greedySet.append(best_id) 
                remset.remove(best_id)  

                current_values[best_id] = np.inf     

                if self.dis_type == "sum":
                    Td_row = sparse_simmat[best_id,:].todense()
                    current_values += np.ravel(Td_row.sum(axis=0))
                        
                elif self.dis_type == "min":
                    Td_row = sparse_simmat[best_id,:].todense()
                    current_values = np.maximum(current_values,np.ravel(Td_row.sum(axis=0)))

                numSelected +=1           
            
            total_greedy_list = greedySet


        return total_greedy_list