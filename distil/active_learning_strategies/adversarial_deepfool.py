import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from torch.autograd import Variable

class AdversarialDeepFool(Strategy):
    def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):
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

        self.model.cpu()
        self.model.eval()
        dis = np.zeros(self.unlabeled_x.shape[0])

        data_pool = self.handler(self.unlabeled_x)
        for i in range(self.unlabeled_x.shape[0]):
            if i % 20 == 0:
                print('adv {}/{}'.format(i, self.unlabeled_x.shape[0]), flush=True)
            x, idx = data_pool[i]
            x = torch.from_numpy(x)
            dis[i] = self.cal_dis(x)

        self.model.to(self.device)
        idxs = dis.argsort()[:budget]
        return idxs


