import numpy as np
import sys
import torch
from queue import PriorityQueue
import torch.nn.functional as F
import apricot
from torch.utils.data import random_split, SequentialSampler, BatchSampler
import math
from collections import defaultdict
import copy
from scipy.sparse import csr_matrix

class SubmodularFunction():

    def __init__(self, device, x_trn, y_trn, model, N_trn, batch_size, if_convex, submod, selection_type):
        self.x_trn = x_trn
        self.y_trn = y_trn
        self.model = model
        self.if_convex = if_convex
        self.device = device
        self.N_trn = N_trn
        self.batch_size = batch_size
        self.submod = submod
        self.selection_type = selection_type

    def distance(self, x, y, exp=2):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def get_index(self, data, data_sub):

        greedyList = []
        for row in data_sub:
            idx_map = np.where(np.all(row == data, axis=1))[0]
            for val in idx_map:
                if val not in greedyList:
                    greedyList.append(val)
                    break

        return greedyList


    def compute_score(self, model_params, idxs):
        self.model.load_state_dict(model_params)
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
                if not self.if_convex:
                    scores_i = F.softmax(self.model(inputs_i), dim=1)
                    y_i = torch.zeros(target_i.size(0), scores_i.size(1)).to(self.device)
                    y_i[range(y_i.shape[0]), target_i] = 1
                    g_is.append(scores_i - y_i)
                else:
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
        self.dist_mat = self.dist_mat.cpu().numpy()

    def lazy_greedy_max(self, budget, model_params):

        classes, no_elements = torch.unique(self.y_trn, return_counts=True)
        len_unique_elements = no_elements.shape[0]
        tem_xtrain = copy.deepcopy(self.x_trn)
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
                    self.compute_score(model_params, idxs)
                    fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=final_per_class_bud[i])
                elif self.submod == 'graph_cut':
                    self.compute_score(model_params, idxs)
                    fl = apricot.functions.graphCut.GraphCutSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=final_per_class_bud[i])
                elif self.submod == 'saturated_coverage':
                    self.compute_score(model_params, idxs)
                    fl = apricot.functions.saturatedCoverage.SaturatedCoverageSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=final_per_class_bud[i])
                elif self.submod == 'sum_redundancy':
                    self.compute_score(model_params, idxs)
                    fl = apricot.functions.sumRedundancy.SumRedundancySelection(random_state=0, metric='precomputed',
                                                                                  n_samples=final_per_class_bud[i])
                elif self.submod == 'feature_based':
                    fl = apricot.functions.featureBased.FeatureBasedSelection(random_state=0, n_samples=final_per_class_bud[i])

                if self.submod == 'feature_based':

                    x_sub = fl.fit_transform(self.x_trn[idxs].numpy())
                    greedyList = self.get_index(self.x_trn[idxs].numpy(), x_sub)
                    total_greedy_list.extend(idxs[greedyList])

                else:  

                    sim_sub = fl.fit_transform(self.dist_mat)
                    greedyList = list(np.argmax(sim_sub, axis=1))
                    total_greedy_list.extend(idxs[greedyList])

        elif self.selection_type == 'Supervised':
            
            
            if self.submod == 'feature_based':
                pass

            else:
                for i in range(len(classes)):
                    
                    if i == 0:
                        idxs = torch.where(self.y_trn == classes[i])[0]
                        N = len(idxs)
                        self.compute_score(model_params, idxs)
                        row = idxs.repeat_interleave(N)
                        col = idxs.repeat(N)
                        data = self.dist_mat.flatten()
                    else:
                        idxs = torch.where(self.y_trn == classes[i])[0]
                        N = len(idxs)
                        self.compute_score(model_params, idxs)
                        row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
                        col = torch.cat((col, idxs.repeat(N)), dim=0)
                        data = np.concatenate([data, self.dist_mat.flatten()], axis=0)
                
                
                sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
                self.dist_mat = sparse_simmat

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


        return total_greedy_list