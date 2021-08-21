  
from .strategy import Strategy
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, ConcatDataset, Dataset

import math

class GLISTER(Strategy):
    
    """
    This is implementation of GLISTER-ACTIVE from the paper GLISTER: Generalization based Data 
    Subset Selection for Efficient and Robust Learning :footcite:`killamsetty2020glister`. GLISTER 
    methods tries to solve a bi-level optimisation problem.
    
    .. math::
        \\overbrace{\\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}} L_V(\\underbrace{\\underset{\\theta}{\\operatorname{argmin\\hspace{0.7mm}}} L_T( \\theta, S)}_{inner-level}, {\\mathcal V})}^{outer-level}
        
    In the above equation, :math:`\\mathcal{U}` denotes the Data without lables i.e. `unlabeled_x`, 
    :math:`\\mathcal{V}` denotes the validation set that guides the subset selection process, :math:`L_T` denotes the
    training loss, :math:`L_V` denotes the validation loss, :math:`S` denotes the data subset selected at each round,  and :math:`k` is the `budget`.
    Since, solving the complete inner-optimization is expensive, GLISTER-ONLINE adopts a online one-step meta approximation where we approximate the solution to inner problem
    by taking a single gradient step.
    The optimization problem after the approximation is as follows:
    
    .. math::
        \\overbrace{\\underset{{S \\subseteq {\\mathcal U}, |S| \\leq k}}{\\operatorname{argmin\\hspace{0.7mm}}} L_V(\\underbrace{\\theta - \\eta \\nabla_{\\theta}L_T(\\theta, S)}_{inner-level}, {\\mathcal V})}^{outer-level}
    
    In the above equation, :math:`\\eta` denotes the step-size used for one-step gradient update.
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled training dataset
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
        - **lr**: The learning rate used for training (float)
    validation_dataset: torch.utils.data.Dataset
        The validation dataset to be used in GLISTER objective
    typeOf: str, optional
        Determines the type of regulariser to be used. Default is **'none'**.
        For random regulariser use **'Rand'**.
        To use Facility Location set functiom as a regulariser use **'FacLoc'**.
        To use Diversity set functiom as a regulariser use **'Diversity'**.
    lam: float, optional
        Determines the amount of regularisation to be applied. Mandatory if is not `typeOf='none'` and by default set to `None`.
        For random regulariser use values should be between 0 and 1 as it determines fraction of points replaced by random points.
        For both 'Diversity' and 'FacLoc', `lam` determines the weightage given to them while computing the gain.
    kernel_batch_size: int, optional
        For 'Diversity' and 'FacLoc' regualrizer versions, similarity kernel is to be computed, which 
        entails creating a 3d torch tensor of dimenssions kernel_batch_size*kernel_batch_size*
        feature dimenssion.Again kernel_batch_size should be such that one can exploit the benefits of 
        tensorization while honouring the resourse constraits. 
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}, validation_dataset = None,
                 typeOf = 'none', lam = None, kernel_batch_size = 200):
        
        super(GLISTER, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    
        self.validation_dataset = validation_dataset
        self.typeOf = typeOf
        self.lam = lam
        self.kernel_batch_size = kernel_batch_size

    def distance(self, x, y, exp = 2):

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
            self.sim_mat = torch.zeros([new_N, new_N], dtype=torch.float32).to(self.device)
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
                #self.sim_mat = const - self.sim_mat

                self.min_dist = (torch.ones(new_N, dtype=torch.float32)*const).to(self.device)

    def _compute_per_element_grads(self):
        
        self.grads_per_elem = self.get_grad_embedding(self.unlabeled_dataset, True)
        self.prev_grads_sum = torch.sum(self.get_grad_embedding(self.labeled_dataset, False), dim=0).view(1, -1)

    def _update_grads_val(self,grads_currX=None, first_init=False):

        embDim = self.model.get_embedding_dim()
        
        if first_init:
            if self.validation_dataset is not None:
                loader = DataLoader(self.validation_dataset,shuffle=False,batch_size=self.args['batch_size'])
                self.out = torch.zeros(len(self.validation_dataset), self.target_classes).to(self.device)
                self.emb = torch.zeros(len(self.validation_dataset), embDim).to(self.device)
            else:
                predicted_y = self.predict(self.unlabeled_dataset).cpu() # Bring to CPU as the loaders used require it
                
                class AddLabelDataset(Dataset):
                    
                    def __init__(self, wrapped_unlabeled_dataset, added_labels):
                        self.wrapped_unlabeled_dataset = wrapped_unlabeled_dataset
                        self.added_labels = added_labels
                        
                    def __getitem__(self, index):
                        unlabeled_data = self.wrapped_unlabeled_dataset[index]
                        label = self.added_labels[index]
                        
                        return unlabeled_data, label
                    
                    def __len__(self):
                        return len(self.wrapped_unlabeled_dataset)
                
                pseudolabeled_dataset = AddLabelDataset(self.unlabeled_dataset, predicted_y)
                
                self.new_dataset = ConcatDataset([pseudolabeled_dataset, self.labeled_dataset])

                loader = DataLoader(self.new_dataset, shuffle=False, batch_size=self.args['batch_size'])
                self.out = torch.zeros(len(self.new_dataset), self.target_classes).to(self.device)
                self.emb = torch.zeros(len(self.new_dataset), embDim).to(self.device)

            self.grads_val_curr = torch.zeros(self.target_classes*(1+embDim), 1).to(self.device)
            
            evaluated_points = 0
            
            with torch.no_grad():

                for x, y in loader:
                    idxs = [iter_index for iter_index in range(evaluated_points, evaluated_points + y.shape[0])]
                    x = x.to(self.device)
                    y = y.to(self.device)
                    init_out, init_l1 = self.model(x,last=True)
                    self.emb[idxs] = init_l1 
                    for j in range(self.target_classes):
                        try:
                            self.out[idxs, j] = init_out[:, j] - (1 * self.args['lr'] * (torch.matmul(init_l1, self.prev_grads_sum[0][(j * embDim) +
                                    self.target_classes:((j + 1) * embDim) + self.target_classes].view(-1, 1)) + self.prev_grads_sum[0][j])).view(-1)
                        except KeyError:
                            raise ValueError("Please pass learning rate used during the training")
                
                    scores = F.softmax(self.out[idxs], dim=1)
                    one_hot_label = torch.zeros(len(y), self.target_classes).to(self.device)
                    one_hot_label.scatter_(1, y.view(-1, 1), 1)
                    l0_grads = scores - one_hot_label
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * init_l1.repeat(1, self.target_classes)

                    self.grads_val_curr += torch.cat((l0_grads, l1_grads), dim=1).sum(dim=0).view(-1, 1)
                    evaluated_points += y.shape[0]
            
            if self.validation_dataset is not None:
                self.grads_val_curr /= len(self.validation_dataset)
                _, self.Y_Val = next(iter(DataLoader(self.validation_dataset, shuffle = False, batch_size = len(self.validation_dataset))))
                self.Y_Val = self.Y_Val.to(self.device)
            else:
                self.grads_val_curr /= predicted_y.shape[0]
                _, self.Y_new = next(iter(DataLoader(self.new_dataset, shuffle = False, batch_size = len(self.new_dataset))))
                self.Y_new = self.Y_new.to(self.device)

        elif grads_currX is not None:
            # update params:
            with torch.no_grad():

                for j in range(self.target_classes):
                    try:
                        self.out[:, j] = self.out[:, j] - (1 * self.args['lr'] * (torch.matmul(self.emb, grads_currX[0][(j * embDim) +
                                    self.target_classes:((j + 1) * embDim) + self.target_classes].view(-1, 1)) +  grads_currX[0][j])).view(-1)
                    except KeyError:
                        print("Please pass learning rate used during the training")

            
                scores = F.softmax(self.out, dim=1)
                if self.validation_dataset is not None:
                    one_hot_label = torch.zeros(self.Y_Val.shape[0], self.target_classes).to(self.device)
                    one_hot_label.scatter_(1,self.Y_Val.view(-1, 1), 1)   
                else:
                    one_hot_label = torch.zeros(self.Y_new.shape[0], self.target_classes).to(self.device)
                    one_hot_label.scatter_(1, self.Y_new.view(-1, 1), 1)
                l0_grads = scores - one_hot_label
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * self.emb.repeat(1, self.target_classes)

                self.grads_val_curr = torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(-1, 1)

    def eval_taylor_modular(self, grads,greedySet=None,remset=None):

        with torch.no_grad():
            if self.typeOf == "FacLoc":
                gains = torch.matmul(grads, self.grads_val_curr) + self.lam*((self.min_dist - \
                    torch.min(self.min_dist,self.sim_mat[remset])).sum(1)).view(-1, 1).to(self.device)
                
            elif self.typeOf == "Diversity" and len(greedySet) > 0:
                gains = torch.matmul(grads, self.grads_val_curr) - \
                    self.lam*self.sim_mat[remset][:, greedySet].sum(1).view(-1, 1).to(self.device)
            else:
                gains = torch.matmul(grads, self.grads_val_curr)
        return gains
    
    def select(self, budget):

        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	

        self.model.eval()

        self._compute_per_element_grads()
        self._update_grads_val(first_init=True)
        
        numSelected = 0
        greedySet = list()
        remainSet = list(range(len(self.unlabeled_dataset)))

        if self.typeOf == 'Rand':
            if self.lam is not None:
                if self.lam >0 and self.lam < 1:
                    curr_bud = (1-self.lam)*budget
                else:
                    raise ValueError("Lambda value should be between 0 and 1")
            else:
                raise ValueError("Please pass a appropriate lambda value for random regularisation")
        else:
            curr_bud = budget

        if self.typeOf == "FacLoc" or self.typeOf == "Diversity":
            if self.lam is not None:
                self._compute_similarity_kernel()
            else:
                if self.typeOf == "FacLoc":
                    raise ValueError("Please pass a appropriate lambda value for Facility Location based regularisation")
                elif self.typeOf == "Diversity":
                    raise ValueError("Please pass a appropriate lambda value for Diversity based regularisation")
        
        while (numSelected < curr_bud):

            if self.typeOf == "Diversity":
                gains = self.eval_taylor_modular(self.grads_per_elem[remainSet],greedySet,remainSet)
            elif self.typeOf == "FacLoc":
                gains = self.eval_taylor_modular(self.grads_per_elem[remainSet],remset=remainSet)
            else:
                gains = self.eval_taylor_modular(self.grads_per_elem[remainSet])#rem_grads)
                
            bestId = remainSet[torch.argmax(gains).item()]
            greedySet.append(bestId)
            remainSet.remove(bestId)
            numSelected += 1
            
            self._update_grads_val(self.grads_per_elem[bestId].view(1, -1))

            if self.typeOf == "FacLoc":
                self.min_dist = torch.min(self.min_dist,self.sim_mat[bestId])
            
        if self.typeOf == 'Rand':
            greedySet.extend(list(np.random.choice(remainSet, size=budget - int(curr_bud),replace=False)))
        
        return greedySet