import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy

def gram_red(L, L_inv, u_loc):
    n = np.shape(L_inv)[0]
    ms = np.array([False for i in range(n)])
    ms[u_loc] = True

    L_red = L[~ms][:, ~ms]

    D = L_inv[~ms][:, ~ms]
    e = L_inv[~ms][:, ms]
    f = L_inv[ms][:, ms]

    L_red_inv = D - e.dot(e.T) / f
    return L_red, L_red_inv

def gram_aug(L_Y, L_Y_inv, b_u, c_u):
    d_u = c_u - b_u.T.dot(L_Y_inv.dot(b_u))
    g_u = L_Y_inv.dot(b_u)

    L_aug = np.block([[L_Y, b_u],[b_u.T, c_u]])
    L_aug_inv = np.block([[L_Y_inv + g_u.dot(g_u.T/d_u), -g_u/d_u], [-g_u.T/d_u, 1.0/d_u]])

    return L_aug, L_aug_inv

def sample_k_imp(Phi, k, max_iter, rng=np.random):
    n = np.shape(Phi)[0]
    Ind = rng.choice(range(n), size=k, replace=False)

    if n == k:
        return Ind

    X = [False] * n
    for i in Ind:
        X[i] = True
    X = np.array(X)

    L_X = Phi[Ind, :].dot(Phi[Ind, :].T)

    L_X_inv = np.linalg.pinv(L_X)

    for i in range(1, max_iter):

        u = rng.choice(np.arange(n)[X])
        v = rng.choice(np.arange(n)[~X])

        for j in range(len(Ind)):
            if Ind[j] == u:
                u_loc = j

        L_Y, L_Y_inv = gram_red(L_X, L_X_inv, u_loc)

        Ind_red = [i for i in Ind if i != u]

        b_u = Phi[Ind_red, :].dot(Phi[[u], :].T)
        c_u = Phi[[u], :].dot(Phi[[u], :].T)
        b_v = Phi[Ind_red, :].dot(Phi[[v], :].T)
        c_v = Phi[[v], :].dot(Phi[[v], :].T)

        p = min(1, (c_v - b_v.T.dot(L_Y_inv.dot(b_v))) / (c_u - b_u.T.dot(L_Y_inv.dot(b_u))) )

        if rng.uniform() <= p:
            X[u] = False
            X[v] = True
            Ind = Ind_red + [v]
            L_X, L_X_inv = gram_aug(L_Y, L_Y_inv, b_v, c_v)

    return Ind

class BaselineSampling(Strategy):
    """
    Implementation of Baseline Sampling Strategy.
    This class extends :class:`active_learning_strategies.strategy.Strategy`
    to include entropy sampling technique to select data points for active learning.

    Parameters
    ----------
    X: numpy array
        Present training/labeled data   
    y: numpy array
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
        Specify optional parameters
        
        batch_size 
        Batch size to be used inside strategy class (int, optional)
    """
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        """
        Constructor method
        """
        super(BaselineSampling, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args={})

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
        gradEmbedding = self.get_grad_embedding(self.unlabeled_x, bias_grad=False).numpy()
        chosen = sample_k_imp(gradEmbedding, budget, max_iter= int(5 * budget * np.log(budget)))
        return chosen
