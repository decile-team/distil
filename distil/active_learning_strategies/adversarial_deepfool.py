import numpy as np
import torch
from .strategy import Strategy
from torch.autograd import Variable

class AdversarialDeepFool(Strategy):
    """
    Implements Adversial Deep Fool Strategy :footcite:`ducoffe2018adversarial`, a Deep-Fool based 
    Active Learning strategy that selects unlabeled samples with the smallest adversarial 
    perturbation. This technique is motivated by the fact that often the distance computation 
    from decision boundary is difficult and intractable for margin-based methods. This 
    technique avoids estimating distance by using Deep-Fool :footcite:`Moosavi-Dezfooli_2016_CVPR` 
    like techniques to estimate how much adversarial perturbation is required to cross the boundary. 
    The smaller the required perturbation, the closer the point is to the boundary.

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

        max_iter
        Maximum Number of Iterations (int, optional)
    """
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
        """
        Constructor method
        """
        if 'max_iter' in args:
            self.max_iter = args['max_iter']
        else:
            self.max_iter = 50        
        super(AdversarialDeepFool, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args={})

    def cal_dis(self, x):

        nx = Variable(torch.unsqueeze(x, 0), requires_grad=True)
        eta = Variable(torch.zeros(nx.shape))

        out = self.model(nx + eta)
        n_class = out.shape[1]
        py = int(out.max(1)[1])
        ny = int(out.max(1)[1])

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(float(fi)) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

            eta += Variable(ri.clone())
            nx.grad.data.zero_()
            out = self.model(nx + eta)
            py = int(out.max(1)[1])
            i_iter += 1

        return (eta*eta).sum()

    def select(self, budget):
        """
        Select next set of points

        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set

        Returns
        ----------
        idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """ 
        self.model.cpu()
        self.model.eval()
        dis = np.zeros(self.unlabeled_x.shape[0])

        data_pool = self.handler(self.unlabeled_x)
        for i in range(self.unlabeled_x.shape[0]):
            # if i % 20 == 0:
            #     print('adv {}/{}'.format(i, self.unlabeled_x.shape[0]), flush=True)
            x, idx = data_pool[i]
            #x = torch.from_numpy(x)
            dis[i] = self.cal_dis(x)

        self.model.to(self.device)
        idxs = dis.argsort()[:budget]
        return idxs


