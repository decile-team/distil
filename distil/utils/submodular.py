import numpy as np
import torch

import apricot

from scipy.sparse import csr_matrix

from .similarity_mat import SimilarityComputation


class SubmodularFunction(SimilarityComputation):

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

    def __init__(self, device, x_trn, y_trn, N_trn, batch_size, submod, selection_type):

        super(SubmodularFunction, self).__init__(device, x_trn, y_trn, N_trn, batch_size)
       
        self.submod = submod
        self.selection_type = selection_type

    def lazy_greedy_max(self, budget):

        """
        Data selection method using different submodular optimization
        functions.
 
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        
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
                
                if self.submod == 'facility_location':
                    self.compute_score(idxs)
                    fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=final_per_class_bud[i])
                elif self.submod == 'graph_cut':
                    self.compute_score(idxs)
                    fl = apricot.functions.graphCut.GraphCutSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=final_per_class_bud[i])
                elif self.submod == 'saturated_coverage':
                    self.compute_score(idxs)
                    fl = apricot.functions.saturatedCoverage.SaturatedCoverageSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=final_per_class_bud[i])
                elif self.submod == 'sum_redundancy':
                    self.compute_score(idxs)
                    fl = apricot.functions.sumRedundancy.SumRedundancySelection(random_state=0, metric='precomputed',
                                                                                  n_samples=final_per_class_bud[i])
                elif self.submod == 'feature_based':
                    fl = apricot.functions.featureBased.FeatureBasedSelection(random_state=0, n_samples=final_per_class_bud[i])

                if self.submod == 'feature_based':

                    x_sub = fl.fit_transform(self.x_trn[idxs].numpy())
                    greedyList = self.get_index(self.x_trn[idxs].numpy(), x_sub)
                    total_greedy_list.extend(idxs[greedyList])

                else:  

                    sim_sub = fl.fit_transform(self.dist_mat.cpu().numpy())
                    greedyList = list(np.argmax(sim_sub, axis=1))
                    total_greedy_list.extend(idxs[greedyList])

        elif self.selection_type == 'Supervised':
            
            
            if self.submod == 'feature_based':
                
                class_map = {}
                for i in range(len_unique_elements):
                    class_map[torch.IntTensor.item(classes[i])] = i #Mapping classes from 0 to n
                    
                sparse_data = torch.zeros([self.x_trn.shape[0], self.x_trn.shape[1]*len_unique_elements])
                for i in range(self.x_trn.shape[0]):
                    
                    start_col = class_map[torch.IntTensor.item(self.y_trn[i])]*self.x_trn.shape[1]
                    end_col = start_col+self.x_trn.shape[1]
                    sparse_data[i, start_col:end_col] = self.x_trn[i, :]

                fl = apricot.functions.featureBased.FeatureBasedSelection(random_state=0, n_samples=budget)
                x_sub = fl.fit_transform(sparse_data.numpy())
                total_greedy_list = self.get_index(sparse_data.numpy(), x_sub)

            else:
                for i in range(len(classes)):
                    
                    if i == 0:
                        idxs = torch.where(self.y_trn == classes[i])[0]
                        N = len(idxs)
                        self.compute_score(idxs)
                        row = idxs.repeat_interleave(N)
                        col = idxs.repeat(N)
                        data = self.dist_mat.cpu().numpy().flatten()
                    else:
                        idxs = torch.where(self.y_trn == classes[i])[0]
                        N = len(idxs)
                        self.compute_score(idxs)
                        row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
                        col = torch.cat((col, idxs.repeat(N)), dim=0)
                        data = np.concatenate([data, self.dist_mat.cpu().numpy().flatten()], axis=0)
                
                
                sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
                #self.dist_mat = sparse_simmat

                if self.submod == 'facility_location':
                    fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=budget)
                elif self.submod == 'graph_cut':
                    fl = apricot.functions.graphCut.GraphCutSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=budget)
                elif self.submod == 'saturated_coverage':
                    fl = apricot.functions.saturatedCoverage.SaturatedCoverageSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=budget)
                elif self.submod == 'sum_redundancy':
                    fl = apricot.functions.sumRedundancy.SumRedundancySelection(random_state=0, metric='precomputed',
                                                                                  n_samples=budget)
                sim_sub = fl.fit_transform(sparse_simmat)
                total_greedy_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))


        if self.selection_type == 'Full':
        

            total_greedy_list = []
            idx_end = self.x_trn.shape[0] - 1
            idxs = torch.linspace(0, idx_end, self.x_trn.shape[0]).long()

            if self.submod == 'facility_location':
                self.compute_score(idxs)
                fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget)
            elif self.submod == 'graph_cut':
                self.compute_score(idxs)
                fl = apricot.functions.graphCut.GraphCutSelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget)
            elif self.submod == 'saturated_coverage':
                self.compute_score(idxs)
                fl = apricot.functions.saturatedCoverage.SaturatedCoverageSelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget)
            elif self.submod == 'sum_redundancy':
                self.compute_score(idxs)
                fl = apricot.functions.sumRedundancy.SumRedundancySelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget)
            elif self.submod == 'feature_based':
                fl = apricot.functions.featureBased.FeatureBasedSelection(random_state=0, n_samples=budget)

            if self.submod == 'feature_based':

                x_sub = fl.fit_transform(self.x_trn.numpy())
                total_greedy_list = self.get_index(self.x_trn.numpy(), x_sub)

            else:  

                sim_sub = fl.fit_transform(self.dist_mat.cpu().numpy())
                total_greedy_list = list(np.argmax(sim_sub, axis=1))

        return total_greedy_list