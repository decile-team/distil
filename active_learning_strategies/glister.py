from .strategy import Strategy
import copy
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

import math
import random

class GLISTER(Strategy):

    def __init__(self,X, Y,unlabeled_x, net, handler, nclasses, args,valid,X_val=None,Y_val=None,\
        loss_criterion=nn.CrossEntropyLoss(),typeOf='none',lam=None,kernel_batch_size = 200): # 
        super(GLISTER, self).__init__(X, Y, unlabeled_x, net, handler,nclasses, args)

        if valid:
            self.X_Val = X_val
            self.Y_Val = Y_val
        self.loss = loss_criterion
        self.valid = valid
        self.typeOf = typeOf
        self.lam = lam
        self.kernel_batch_size = kernel_batch_size
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def distance(self,x, y, exp = 2):

      n = x.size(0)
      m = y.size(0)
      d = x.size(1)

      x = x.unsqueeze(1).expand(n, m, d)
      y = y.unsqueeze(0).expand(n, m, d)

      if self.typeOf == "FacLoc":
          dist = torch.pow(x - y, exp).sum(2) 
      elif self.typeOf == "Diversity":
          dist = torch.exp((-1 * torch.pow(x - y, exp).sum(2))/2)
      
      return dist 

    def _compute_similarity_kernel(self):
        
        g_is = []
        for item in range(math.ceil(len(self.grads_per_elem) / self.kernel_batch_size)):
            inputs = self.grads_per_elem[item *self.kernel_batch_size:(item + 1) *self.kernel_batch_size]
            g_is.append(inputs)

        with torch.no_grad():
            
            new_N = len(self.grads_per_elem)
            self.sim_mat = torch.zeros([new_N, new_N], dtype=torch.float32)#.to(self.device)
            first_i = True
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.sim_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j)

            if self.typeOf == "FacLoc":
                const = torch.max(self.sim_mat).item()
                self.sim_mat = const - self.sim_mat

                self.max_sim = torch.zeros(new_N, dtype=torch.float32)#.to(self.device)

    def _compute_per_element_grads(self):
        
        self.grads_per_elem = self.get_grad_embedding(self.unlabeled_x)
        self.prev_grads_sum = torch.sum(self.get_grad_embedding(self.X,self.Y),dim=0)

    def _update_grads_val(self,grads_currX=None, first_init=False):

        embDim = self.model.get_embedding_dim()
        
        if first_init:
            if self.valid:
                if self.X_Val is not None:
                    loader = DataLoader(self.handler(self.X_Val,self.Y_Val,select=False),shuffle=False,\
                        batch_size=self.args['batch_size'])
                    self.out = torch.zeros(self.Y_Val.shape[0], self.target_classes).to(self.device)
                    self.emb = torch.zeros(self.Y_Val.shape[0], embDim).to(self.device)
                else:
                    raise ValueError("Since Valid is set True, please pass a appropriate Validation set")
            
            else:
                predicted_y = self.predict(self.unlabeled_x)
                loader = DataLoader(self.handler(self.unlabeled_x,predicted_y,select=False),shuffle=False,\
                    batch_size=self.args['batch_size'])
                self.out = torch.zeros(predicted_y.shape[0], self.target_classes).to(self.device)
                self.emb = torch.zeros(self.Y_Val.shape[0], embDim).to(self.device)

                self.grads_val_curr = torch.zeros(self.target_classes*(1+embDim), 1).to(self.device)
            
            with torch.no_grad():

                for x, y, idxs in loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    init_out, init_l1 = self.model(x)
                    self.emb[idxs] = init_l1 
                    for j in range(self.target_classes):
                        self.out[idxs, j] = init_out[:, j] - (1 * self.args['lr'] * (torch.matmul(init_l1, self.prev_grads_sum[0][(j * embDim) +
                                    self.target_classes:((j + 1) * embDim) + self.target_classes].view(-1, 1)) + self.prev_grads_sum[0][j])).view(-1)
                
                    scores = F.softmax(self.out, dim=1)
                    one_hot_label = torch.zeros(len(y), self.target_classes).to(self.device)
                    one_hot_label.scatter_(1, y.view(-1, 1), 1)
                    l0_grads = scores - one_hot_label
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * init_l1.repeat(1, self.target_classes)

                    self.grads_val_curr += torch.cat((l0_grads, l1_grads), dim=1).sum(dim=0).view(-1, 1)
            
            if self.valid:
                self.grads_val_curr /= self.Y_Val.shape[0]
            else:
                self.grads_val_curr /= predicted_y.shape[0]

        elif grads_currX is not None:
            # update params:
            with torch.no_grad():

                for j in range(self.target_classes):
                    self.out[:, j] = self.out[:, j] - (1 * self.args['lr'] * (torch.matmul(self.emb, grads_currX[0][(j * embDim) +
                                self.target_classes:((j + 1) * embDim) + self.target_classes].view(-1, 1)) +  grads_currX[0][j])).view(-1)
            
                scores = F.softmax(self.out, dim=1)
                one_hot_label = torch.zeros(len(y), self.target_classes).to(self.device)
                one_hot_label.scatter_(1, y.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.emb.repeat(1, self.target_classes)

                self.grads_val_curr = torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads,greedySet=None,remset=None):

        with torch.no_grad():
            if self.typeOf == "FacLoc":
                gains = torch.matmul(grads, self.grads_val_curr) + self.lam*(torch.max(self.max_sim,\
                    self.sim_mat)- self.max_sim).sum().to(self.device)
            elif self.typeOf == "Diversity" and len(greedySet) > 0:
                gains = torch.matmul(grads, self.grads_val_curr) - \
                    self.lam*self.sim_mat[remset][:, greedySet].sum(1).view(-1, 1).to(self.device)
            else:
                gains = torch.matmul(grads, self.grads_val_curr)
        return gains
    
    def select(self, n):

        self._compute_per_element_grads()
        self._update_grads_val(first_init=True)
        
        numSelected = 0
        greedySet = list()
        remainSet = list(range(self.unlabeled_x.shape[0]))

        if self.typeOf == 'Rand':
            if self.lam is not None:
                if self.lam >0 and self.lam < 1:
                    budget = (1-self.lam)*n
                else:
                    raise ValueError("Lambda value should be between 0 and 1")
            else:
                raise ValueError("Please pass a appropriate lambda value for random regularisation")
        else:
            budget = n

        if self.typeOf == "FacLoc" and self.typeOf == "Diversity":
            if self.lam is not None:
                self._compute_similarity_kernel()
            else:
                if self.typeOf == "FacLoc":
                    raise ValueError("Please pass a appropriate lambda value for Facility Location based regularisation")
                elif self.typeOf == "Diversity":
                    raise ValueError("Please pass a appropriate lambda value for Diversity based regularisation")
        
        while (numSelected < budget):

            if self.typeOf == "Diversity":
                gains = self.eval_taylor_modular(self.grads_per_elem[remainSet],greedySet,remainSet)
            else:
                gains = self.eval_taylor_modular(self.grads_per_elem[remainSet])#rem_grads)
                
            bestId = remainSet[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            numSelected += 1
            
            self._update_grads_val(self.grads_per_elem[bestId].view(1, -1))

            if self.typeOf == "FacLoc":
                self.max_sim = torch.max(self.max_sim,self.sim_mat[bestId])
            
        if self.typeOf == 'Rand':
            greedySet.extend(list(np.random.choice(remainSet, size=n - int(budget),replace=False)))
        
        return greedySet