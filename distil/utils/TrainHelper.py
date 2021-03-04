import pandas as pd 
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('../')
# from distil.active_learning_strategies import FASS, EntropySampling, EntropySamplingDropout, RandomSampling,\
#                                 LeastConfidence,LeastConfidenceDropout, MarginSampling, MarginSamplingDropout, \
#                                 CoreSet, GLISTER, BADGE, AdversarialBIM, AdversarialDeepFool, KMeansSampling, BaselineSampling, \
#                                   BALDDropout      

from distil.active_learning_strategies import BALDDropout
from distil.utils.models.logreg_net import LogisticRegNet
from distil.utils.models.simpleNN_net import TwoLayerNet
from distil.utils.DataHandler import DataHandler_Points


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

#custom training
class data_train:

    def __init__(self, X, Y, net, handler, args):

        self.X = X
        self.Y = Y
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.use_cuda = torch.cuda.is_available()
        
        if 'islogs' not in args:
            self.args['islogs'] = False

    def update_index(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def update_data(self, X, Y):
    	self.X = X
    	self.Y = Y

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            if self.use_cuda:
                x, y = Variable(x.cuda()), Variable(y.cuda())
            else:
                x, y = Variable(x), Variable(y)
            optimizer.zero_grad()
            out = self.clf(x)
            loss = F.cross_entropy(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset.X), loss

    
    def train(self):

        print('Training..')
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        train_logs = []
        n_epoch = self.args['n_epoch']
        if self.use_cuda:
            self.clf =  self.net.apply(weight_reset).cuda()
        else:
            self.clf =  self.net.apply(weight_reset)

        optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        loader_tr = DataLoader(self.handler(self.X, self.Y, False), batch_size=batch_size)
        epoch = 1
        accCurrent = 0
        while accCurrent < 0.95 and epoch < n_epoch: 
            accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            epoch += 1
            print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
            log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
            train_logs.append(log_string)
            if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        print('Epoch:', str(epoch),'Training accuracy:',round(accCurrent, 3), flush=True)
        if self.args['islogs']:
            return self.clf, train_logs
        else:
            return self.clf