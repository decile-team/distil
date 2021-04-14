import apricot
import math
import numpy as np
import torch

from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler

from .calculate_class_budgets import calculate_class_budgets
from .gradmatch_solvers import OrthogonalMP_REG_Parallel, Fixed_Weight_Greedy_Parallel

class DataSelectionStrategy(object):
    """
    Implementation of Data Selection Strategy class which serves as base class for other
    dataselectionstrategies for general learning frameworks.
    Parameters
        ----------
        trainloader: class
            Loading the training data using pytorch dataloader
        valloader: class
            Loading the validation data using pytorch dataloader
        model: class
            Model architecture used for training
        num_classes: int
            Number of target classes in the dataset
        linear_layer: bool
            If True, we use the last fc layer weights and biases gradients
            If False, we use the last fc layer biases gradients
        loss: class
            PyTorch Loss function
    """

    def __init__(self, trainloader, valloader, model, num_classes, linear_layer, loss, device):
        """
        Constructer method
        """
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader
        self.model = model
        self.N_trn = len(trainloader.sampler)
        self.N_val = len(valloader.sampler)
        self.grads_per_elem = None
        self.val_grads_per_elem = None
        self.numSelected = 0
        self.linear_layer = linear_layer
        self.num_classes = num_classes
        self.trn_lbls = None
        self.val_lbls = None
        self.loss = loss
        self.device = device

    def select(self, budget, model_params):
        pass

    def get_labels(self, valid=False):
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                self.trn_lbls = targets.view(-1, 1)
            else:
                self.trn_lbls = torch.cat((self.trn_lbls, targets.view(-1, 1)), dim=0)
        self.trn_lbls = self.trn_lbls.view(-1)

        if valid:
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                if batch_idx == 0:
                    self.val_lbls = targets.view(-1, 1)
                else:
                    self.val_lbls = torch.cat((self.val_lbls, targets.view(-1, 1)), dim=0)
            self.val_lbls = self.val_lbls.view(-1)

    def compute_gradients(self, valid=False, batch=False, perClass=False):
        """
        Computes the gradient of each element.
        Here, the gradients are computed in a closed form using CrossEntropyLoss with reduction set to 'none'.
        This is done by calculating the gradients in last layer through addition of softmax layer.
        Using different loss functions, the way we calculate the gradients will change.
        For LogisticLoss we measure the Mean Absolute Error(MAE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:
        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left| x_n - y_n \\right|,
        where :math:`N` is the batch size.
        For MSELoss, we measure the Mean Square Error(MSE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:
        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left( x_n - y_n \\right)^2,
        where :math:`N` is the batch size.
        Parameters
        ----------
        valid: bool
            if True, the function also computes the validation gradients
        batch: bool
            if True, the function computes the gradients of each mini-batch
        perClass: bool
            if True, the function computes the gradients using perclass dataloaders
        """
        if perClass:
            embDim = self.model.get_embedding_dim()
            for batch_idx, (inputs, targets) in enumerate(self.pctrainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if batch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                    if batch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            torch.cuda.empty_cache()
            if self.linear_layer:
                self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.grads_per_elem = l0_grads

            if valid:
                for batch_idx, (inputs, targets) in enumerate(self.pcvalloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    if batch_idx == 0:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        loss = self.loss(out, targets).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        if batch:
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    else:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        loss = self.loss(out, targets).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                        if batch:
                            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                torch.cuda.empty_cache()
                if self.linear_layer:
                    self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    self.val_grads_per_elem = l0_grads
        else:
            embDim = self.model.get_embedding_dim()
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if batch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                    if batch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

            torch.cuda.empty_cache()

            if self.linear_layer:
                self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.grads_per_elem = l0_grads
            if valid:
                for batch_idx, (inputs, targets) in enumerate(self.valloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                    if batch_idx == 0:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        loss = self.loss(out, targets).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                        if batch:
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    else:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        loss = self.loss(out, targets).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                        if batch:
                            batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
                torch.cuda.empty_cache()
                if self.linear_layer:
                    self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    self.val_grads_per_elem = l0_grads

    def update_model(self, model_params):
        """
        Update the models parameters
        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)

class OMPGradMatchStrategy(DataSelectionStrategy):
    """
    Implementation of OMPGradMatch Strategy from the paper :footcite:`sivasubramanian2020gradmatch` for supervised learning frameworks.
    OMPGradMatch strategy tries to solve the optimization problem given below:
    .. math::
        \\min_{\\mathbf{w}, S: |S| \\leq k} \\Vert \\sum_{i \\in S} w_i \\nabla_{\\theta}L_T^i(\\theta) -  \\nabla_{\\theta}L(\\theta)\\Vert
    In the above equation, :math:`\\mathbf{w}` denotes the weight vector that contains the weights for each data instance, :math:`\mathcal{U}` training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`L` denotes either training loss or validation loss depending on the parameter valid,
    :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.
    The above optimization problem is solved using the Orthogonal Matching Pursuit(OMP) algorithm.
    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    eta: float
        Learning rate. Step size for the one step gradient update
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    selection_type: str
        Type of selection -
        - 'PerClass': PerClass method is where OMP algorithm is applied on each class data points seperately.
        - 'PerBatch': PerBatch method is where OMP algorithm is applied on each minibatch data points.
        - 'PerClassPerGradient': PerClassPerGradient method is same as PerClass but we use the gradient corresponding to classification layer of that class only.
    valid : bool, optional
        If valid==True we use validation dataset gradient sum in OMP otherwise we use training dataset (default: False)
    lam : float
        Regularization constant of OMP solver
    eps : float
        Epsilon parameter to which the above optimization problem is solved using OMP algorithm
    """

    def __init__(self, trainloader, valloader, model, loss_type,
                 eta, device, num_classes, linear_layer, selection_type, valid=True, lam=0, eps=1e-4, r=1):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss_type, device)
        self.loss_type = loss_type
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.valid = valid
        self.lam = lam
        self.eps = eps

    def ompwrapper(self, X, Y, bud):
        reg = OrthogonalMP_REG_Parallel(X, Y, nnz=bud,
                                          positive=True, lam=self.lam,
                                          tol=self.eps, device=self.device)
        ind = torch.nonzero(reg).view(-1)
        return ind.tolist(), reg[ind].tolist()

    def select(self, budget, model_params):
        """
        Apply OMP Algorithm for data selection
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        Returns
        ----------
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """
        self.update_model(model_params)

        if self.selection_type == 'PerClass':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            
            # Calculate the class budgets to be used.
            class_budgets = calculate_class_budgets(budget, self.num_classes, self.trn_lbls, self.N_trn)

            for i in range(self.num_classes):
                if class_budgets[i] == 0:
                    print("SKIPPING CLASS", i, "AS THERE IS NO BUDGET")
                    continue

                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                          shuffle=False, pin_memory=True)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                    shuffle=False, pin_memory=True)

                self.compute_gradients(self.valid, batch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                if self.valid:
                    sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)
                idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                          sum_val_grad, class_budgets[i])
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)

        elif self.selection_type == 'PerBatch':
            self.compute_gradients(self.valid, batch=True, perClass=False)
            idxs = []
            gammas = []
            trn_gradients = self.grads_per_elem
            if self.valid:
                sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
            else:
                sum_val_grad = torch.sum(trn_gradients, dim=0)
            idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                     sum_val_grad, math.ceil(budget/self.trainloader.batch_size))
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(idxs_temp)):
                tmp = batch_wise_indices[idxs_temp[i]]
                idxs.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))

        diff = budget - len(idxs)

        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])
            idxs = np.array(idxs)
            gammas = np.array(gammas)

        if self.selection_type in ["PerClass"]:
            rand_indices = np.random.permutation(len(idxs))
            idxs = list(np.array(idxs)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])

        return idxs, gammas
    
class FixedWeightGradMatchStrategy(DataSelectionStrategy):
    """
    Implementation of OMPGradMatch Strategy from the paper :footcite:`sivasubramanian2020gradmatch` for supervised learning frameworks.
    OMPGradMatch strategy tries to solve the optimization problem given below:
    .. math::
        \\min_{\\mathbf{w}, S: |S| \\leq k} \\Vert \\sum_{i \\in S} w_i \\nabla_{\\theta}L_T^i(\\theta) -  \\nabla_{\\theta}L(\\theta)\\Vert
    In the above equation, :math:`\\mathbf{w}` denotes the weight vector that contains the weights for each data instance, :math:`\mathcal{U}` training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`L` denotes either training loss or validation loss depending on the parameter valid,
    :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.
    The above optimization problem is solved using the Orthogonal Matching Pursuit(OMP) algorithm.
    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    eta: float
        Learning rate. Step size for the one step gradient update
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    selection_type: str
        Type of selection -
        - 'PerClass': PerClass method is where OMP algorithm is applied on each class data points seperately.
        - 'PerBatch': PerBatch method is where OMP algorithm is applied on each minibatch data points.
        - 'PerClassPerGradient': PerClassPerGradient method is same as PerClass but we use the gradient corresponding to classification layer of that class only.
    valid : bool, optional
        If valid==True we use validation dataset gradient sum in OMP otherwise we use training dataset (default: False)
    lam : float
        Regularization constant of OMP solver
    eps : float
        Epsilon parameter to which the above optimization problem is solved using OMP algorithm
    """

    def __init__(self, trainloader, valloader, model, loss_type,
                 eta, device, num_classes, linear_layer, selection_type, valid=True, r=1):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss_type, device)
        self.loss_type = loss_type
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.valid = valid

    def fixed_weight_wrapper(self, X, Y, val_set_size, bud):
        reg = Fixed_Weight_Greedy_Parallel(X, Y, val_set_size, nnz=bud, device=self.device)
        ind = torch.nonzero(reg).view(-1)
        return ind.tolist()

    def select(self, budget, model_params):
        """
        Apply OMP Algorithm for data selection
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        Returns
        ----------
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """
        self.update_model(model_params)

        if self.selection_type == 'PerClass':
            self.get_labels(valid=self.valid)
            idxs = []

            # Calculate the class budgets to be used.
            class_budgets = calculate_class_budgets(budget, self.num_classes, self.trn_lbls, self.N_trn)

            for i in range(self.num_classes):
                if class_budgets[i] == 0:
                    print("SKIPPING CLASS", i, "AS THERE IS NO BUDGET")
                    continue

                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()

                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                          shuffle=False, pin_memory=True)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                    shuffle=False, pin_memory=True)

                self.compute_gradients(self.valid, batch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                if self.valid:
                    val_set_size = self.val_grads_per_elem.shape[0]
                    sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
                else:
                    val_set_size = self.trn_gradients.shape[0]
                    sum_val_grad = torch.sum(trn_gradients, dim=0)
                idxs_temp = self.fixed_weight_wrapper(torch.transpose(trn_gradients, 0, 1),
                                          sum_val_grad, val_set_size, class_budgets[i])

                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
        
        elif self.selection_type == 'PerBatch':
            self.compute_gradients(self.valid, batch=True, perClass=False)
            idxs = []
            
            trn_gradients = self.grads_per_elem
            if self.valid:
                val_set_size = self.val_grads_per_elem.shape[0]
                sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
            else:
                val_set_size = self.trn_gradients.shape[0]
                sum_val_grad = torch.sum(trn_gradients, dim=0)
            idxs_temp = self.fixed_weight_wrapper(torch.transpose(trn_gradients, 0, 1),
                                                     sum_val_grad, val_set_size, math.ceil(budget/self.trainloader.batch_size))
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(idxs_temp)):
                tmp = batch_wise_indices[idxs_temp[i]]
                idxs.extend(tmp)

        # Account for the labeled set weights/indices by adding labeled_set_size to budget.
        diff = budget - len(idxs)

        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            idxs = np.array(idxs)

        if self.selection_type in ["PerClass"]:
            rand_indices = np.random.permutation(len(idxs))
            idxs = list(np.array(idxs)[rand_indices])

        return idxs

class CRAIGStrategy(DataSelectionStrategy):
    """
    Implementation of CRAIG Strategy from the paper :footcite:`mirzasoleiman2020coresets` for supervised learning frameworks.
    CRAIG strategy tries to solve the optimization problem given below for convex loss functions:
    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| x^i - x^j \\|
    In the above equation, :math:`\\mathcal{U}` denotes the training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.
    Since, the above optimization problem is not dependent on model parameters, we run the subset selection only once right before the start of the training.
    CRAIG strategy tries to solve the optimization problem given below for non-convex loss functions:
    .. math::
        \\sum_{i\\in \\mathcal{U}} \\min_{j \\in S, |S| \\leq k} \\| \\nabla_{\\theta} {L_T}^i(\\theta) - \\nabla_{\\theta} {L_T}^j(\\theta) \\|
    In the above equation, :math:`\\mathcal{U}` denotes the training set, :math:`L_T` denotes the training loss, :math:`S` denotes the data subset selected at each round,
    and :math:`k` is the budget for the subset. In this case, CRAIG acts an adaptive subset selection strategy that selects a new subset every epoch.
    Both the optimization problems given above are an instance of facility location problems which is a submodular function. Hence, it can be optimally solved using greedy selection methods.
    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss_type: class
        The type of loss criterion
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    if_convex: bool
        If convex or not
    selection_type: str
        Type of selection:
         - 'PerClass': PerClass Implementation where the facility location problem is solved for each class seperately for speed ups.
         - 'Supervised':  Supervised Implementation where the facility location problem is solved using a sparse similarity matrix by assigning the similarity of a point with other points of different class to zero.
    """

    def __init__(self, trainloader, valloader, model, loss,
                 device, num_classes, linear_layer, if_convex, selection_type, optimizer='lazy'):
        """
        Constructer method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device)
        self.if_convex = if_convex
        self.selection_type = selection_type
        self.optimizer = optimizer

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
        dist = torch.pow(x - y, exp).sum(2)
        # dist = torch.exp(-1 * torch.pow(x - y, 2).sum(2))
        return dist

    def compute_score(self, model_params, idxs):
        """
        Compute the score of the indices.
        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        idxs: list
            The indices
        """

        trainset = self.trainloader.sampler.data_source
        subset_loader = torch.utils.data.DataLoader(trainset, batch_size=self.trainloader.batch_size, shuffle=False,
                                                    sampler=SubsetRandomSampler(idxs),
                                                    pin_memory=True)
        self.model.load_state_dict(model_params)
        self.N = 0
        g_is = []

        if self.if_convex:
            for batch_idx, (inputs, targets) in enumerate(subset_loader):
                inputs, targets = inputs, targets
                if self.selection_type == 'PerBatch':
                    self.N += 1
                    g_is.append(inputs.view(inputs.size()[0], -1).mean(dim=0).view(1, -1))
                else:
                    self.N += inputs.size()[0]
                    g_is.append(inputs.view(inputs.size()[0], -1))
        else:
            embDim = self.model.get_embedding_dim()
            for batch_idx, (inputs, targets) in enumerate(subset_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if self.selection_type == 'PerBatch':
                    self.N += 1
                else:
                    self.N += inputs.size()[0]
                out, l1 = self.model(inputs, freeze=True, last=True)
                loss = self.loss(out, targets).sum()
                l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if self.selection_type == 'PerBatch':
                        g_is.append(torch.cat((l0_grads, l1_grads), dim=1).mean(dim=0).view(1, -1))
                    else:
                        g_is.append(torch.cat((l0_grads, l1_grads), dim=1))
                else:
                    if self.selection_type == 'PerBatch':
                        g_is.append(l0_grads.mean(dim=0).view(1, -1))
                    else:
                        g_is.append(l0_grads)

        self.dist_mat = torch.zeros([self.N, self.N], dtype=torch.float32)
        first_i = True
        if self.selection_type == 'PerBatch':
            g_is = torch.cat(g_is, dim=0)
            self.dist_mat = self.distance(g_is, g_is).cpu()
        else:
            for i, g_i in enumerate(g_is, 0):
                if first_i:
                    size_b = g_i.size(0)
                    first_i = False
                for j, g_j in enumerate(g_is, 0):
                    self.dist_mat[i * size_b: i * size_b + g_i.size(0),
                    j * size_b: j * size_b + g_j.size(0)] = self.distance(g_i, g_j).cpu()
        self.const = torch.max(self.dist_mat).item()
        self.dist_mat = (self.const - self.dist_mat).numpy()

    def compute_gamma(self, idxs):
        """
        Compute the gamma values for the indices.
        Parameters
        ----------
        idxs: list
            The indices
        Returns
        ----------
        gamma: list
            Gradient values of the input indices
        """

        if self.selection_type in ['PerClass', 'PerBatch']:
            gamma = [0 for i in range(len(idxs))]
            best = self.dist_mat[idxs]  # .to(self.device)
            rep = np.argmax(best, axis=0)
            for i in rep:
                gamma[i] += 1
        elif self.selection_type == 'Supervised':
            gamma = [0 for i in range(len(idxs))]
            best = self.dist_mat[idxs]  # .to(self.device)
            rep = np.argmax(best, axis=0)
            for i in range(rep.shape[1]):
                gamma[rep[0, i]] += 1
        return gamma

    def get_similarity_kernel(self):
        """
        Obtain the similarity kernel.
        Returns
        ----------
        kernel: ndarray
            Array of kernel values
        """
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        kernel = np.zeros((labels.shape[0], labels.shape[0]))
        for target in np.unique(labels):
            x = np.where(labels == target)[0]
            # prod = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
            for i in x:
                kernel[i, x] = 1
        return kernel

    def select(self, budget, model_params):
        """
        Data selection method using different submodular optimization
        functions.
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters
        optimizer: str
            The optimization approach for data selection. Must be one of
            'random', 'modular', 'naive', 'lazy', 'approximate-lazy', 'two-stage',
            'stochastic', 'sample', 'greedi', 'bidirectional'
        Returns
        ----------
        total_greedy_list: list
            List containing indices of the best datapoints
        gammas: list
            List containing gradients of datapoints present in greedySet
        """

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if batch_idx == 0:
                labels = targets
            else:
                tmp_target_i = targets
                labels = torch.cat((labels, tmp_target_i), dim=0)
        # per_class_bud = int(budget / self.num_classes)
        total_greedy_list = []
        gammas = []
        if self.selection_type == 'PerClass':
            self.get_labels(False)
            class_budgets = calculate_class_budgets(budget, self.num_classes, self.trn_lbls, self.N_trn)
            
            for i in range(self.num_classes):
                idxs = torch.where(labels == i)[0]
                self.compute_score(model_params, idxs)
                fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                                  n_samples=class_budgets[i],
                                                                                  optimizer=self.optimizer)
                sim_sub = fl.fit_transform(self.dist_mat)
                greedyList = list(np.argmax(sim_sub, axis=1))
                gamma = self.compute_gamma(greedyList)
                total_greedy_list.extend(idxs[greedyList])
                gammas.extend(gamma)
            rand_indices = np.random.permutation(len(total_greedy_list))
            total_greedy_list = list(np.array(total_greedy_list)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])
        elif self.selection_type == 'Supervised':
            for i in range(self.num_classes):
                if i == 0:
                    idxs = torch.where(labels == i)[0]
                    N = len(idxs)
                    self.compute_score(model_params, idxs)
                    row = idxs.repeat_interleave(N)
                    col = idxs.repeat(N)
                    data = self.dist_mat.flatten()
                else:
                    idxs = torch.where(labels == i)[0]
                    N = len(idxs)
                    self.compute_score(model_params, idxs)
                    row = torch.cat((row, idxs.repeat_interleave(N)), dim=0)
                    col = torch.cat((col, idxs.repeat(N)), dim=0)
                    data = np.concatenate([data, self.dist_mat.flatten()], axis=0)
            sparse_simmat = csr_matrix((data, (row.numpy(), col.numpy())), shape=(self.N_trn, self.N_trn))
            self.dist_mat = sparse_simmat
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=budget, optimizer=self.optimizer)
            sim_sub = fl.fit_transform(sparse_simmat)
            total_greedy_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            gammas = self.compute_gamma(total_greedy_list)
        elif self.selection_type == 'PerBatch':
            idxs = torch.arange(self.N_trn)
            N = len(idxs)
            self.compute_score(model_params, idxs)
            fl = apricot.functions.facilityLocation.FacilityLocationSelection(random_state=0, metric='precomputed',
                                                                              n_samples=math.ceil(
                                                                                  budget / self.trainloader.batch_size),
                                                                              optimizer=self.optimizer)
            sim_sub = fl.fit_transform(self.dist_mat)
            temp_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
            gammas_temp = self.compute_gamma(temp_list)
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(temp_list)):
                tmp = batch_wise_indices[temp_list[i]]
                total_greedy_list.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))
        return total_greedy_list, gammas

# Define a GradMatch Active Select handler
class SupervisedSelectHandler(Dataset):
    
    def __init__(self, wrapped_handler):
        
        self.wrapped_handler = wrapped_handler
        
    def __getitem__(self, index):
        
        if self.wrapped_handler.select == False:
            
            x, y, index = self.wrapped_handler.__getitem__(index)
            return x, y
        
        else:
        
            x, index = self.wrapped_handler.__getitem__(index)
            return x
        
    def __len__(self):
        
        return len(self.wrapped_handler.X)