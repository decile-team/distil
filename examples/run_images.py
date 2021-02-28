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
from distil.active_learning_strategies import FASS, EntropySampling, EntropySamplingDropout, RandomSampling,\
                                LeastConfidence,LeastConfidenceDropout, MarginSampling, MarginSamplingDropout, \
                                CoreSet, BADGE
# from models.linearmodel import mlpMod, linMod, ResNet18
# from models.linearmodel import linMod
# from models.mlpmod import mlpMod
# from models.resnet import ResNet18
from distil.utils.models.cifar10net import CifarNet
from distil.utils.models.mnist_net import MnistNet
from distil.utils.DataHandler import DataHandler_MNIST, DataHandler_CIFAR10
from distil.utils.dataset import get_dataset
from distil.utils.TrainHelper import data_train

#custom training
# class data_train:

#     def __init__(self, X, Y, net, handler, args):

#         self.X = X
#         self.Y = Y
#         self.net = net
#         self.handler = handler
#         self.args = args
#         self.n_pool = len(Y)
#         self.use_cuda = torch.cuda.is_available()

#     def update_index(self, idxs_lb):
#         self.idxs_lb = idxs_lb

#     def update_data(self, X, Y):
#     	self.X = X
#     	self.Y = Y

#     def _train(self, epoch, loader_tr, optimizer):
#         self.clf.train()
#         accFinal = 0.

#         for batch_id, (x, y, idxs) in enumerate(loader_tr):
#             if self.use_cuda:
#                 x, y = Variable(x.cuda()), Variable(y.cuda())
#             else:
#                 x, y = Variable(x), Variable(y)
#             optimizer.zero_grad()
#             out = self.clf(x)
#             loss = F.cross_entropy(out, y.long())
#             accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
#             loss.backward()

#             # clamp gradients, just in case
#             # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

#             optimizer.step()
#         return accFinal / len(loader_tr.dataset.X)

    
#     def train(self):

#         print('Training..')
#         def weight_reset(m):
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 m.reset_parameters()

#         n_epoch = self.args['n_epoch']
#         if self.use_cuda:
#             self.clf =  self.net.apply(weight_reset).cuda()
#         else:
#             self.clf =  self.net.apply(weight_reset)

#         optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
#         loader_tr = DataLoader(self.handler(self.X, self.Y, False), shuffle=True, batch_size = args['batch_size'])
#         epoch = 1
#         accCurrent = 0
#         while accCurrent < 0.95 and epoch < n_epoch: 
#             accCurrent = self._train(epoch, loader_tr, optimizer)
#             epoch += 1
#             print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
            
#             if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
#                 self.clf = self.net.apply(weight_reset)
#                 optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

#         print('Epoch:', str(epoch),'Training accuracy:',round(accCurrent, 3), flush=True)
#         return self.clf

data_set_name = 'CIFAR10'
download_path = '../downloaded_data/'
X, y, X_test, y_test = get_dataset(data_set_name, download_path)
dim = np.shape(X)[1:]
handler = DataHandler_CIFAR10

print(type(X), type(y), type(X_test), type(y_test))
X_tr = X[:2000]
y_tr = y[:2000]
X_unlabeled = X[2000:]
y_unlabeled = y[2000:]

X_test = X_test
y_test = y_test.numpy()

nclasses = 10
n_rounds = 11    ##Number of rounds to run active learning
budget = 10 
print('Nclasses ', nclasses)

# net = ResNet18(channel=1)
# net = mlpMod(dim, nclasses, embSize=24)
net = CifarNet()
train_args = {'n_epoch':10, 'lr':float(0.001), 'batch_size':16} 
strategy_args = {'batch_size' : 64}
strategy = BADGE(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)

# strategy_args = {'batch_size' : 1, 'submod' : 'facility_location', 'selection_type' : 'PerClass'} 
# strategy = FASS(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)

# strategy_args = {'batch_size' : 16}
# strategy = EntropySampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses)
# strategy = RandomSampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
# strategy = LeastConfidence(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = MarginSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses)

# strategy_args = {'batch_size' : 2, 'n_drop' : 2}
# strategy = EntropySamplingDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = LeastConfidenceDropout(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
# strategy = MarginSamplingDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

# strategy_args = {'batch_size' : 16, 'tor':1e-4}
# strategy = CoreSet(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)

#Training first set of points
dt = data_train(X_tr, y_tr, net, handler, train_args)
clf = dt.train()
strategy.update_model(clf)
y_pred = strategy.predict(X_test).numpy()

acc = np.zeros(n_rounds)
acc[0] = (1.0*(y_test == y_pred)).sum().item() / len(y_test)
print('Initial Testing accuracy:', round(acc[0], 3), flush=True)

##User Controlled Loop
for rd in range(1, n_rounds):
    print('-------------------------------------------------')
    print('Round', rd) 
    print('-------------------------------------------------')
    idx = strategy.select(budget)
    print('New data points added -', len(idx))
    strategy.save_state()

    #Adding new points to training set
    X_tr = np.concatenate((X_tr, X_unlabeled[idx]), axis=0)
    X_unlabeled = np.delete(X_unlabeled, idx, axis = 0)

    #Human In Loop, Assuming user adds new labels here
    y_tr = np.concatenate((y_tr, y_unlabeled[idx]), axis = 0)
    y_unlabeled = np.delete(y_unlabeled, idx, axis = 0)
    print('Number of training points -',X_tr.shape[0])
    print('Number of labels -', y_tr.shape[0])
    print('Number of unlabeled points -', X_unlabeled.shape[0])

    #Reload state and start training
    strategy.load_state()
    strategy.update_data(X_tr, y_tr, X_unlabeled)
    dt.update_data(X_tr, y_tr)

    clf = dt.train()
    strategy.update_model(clf)
    y_pred = strategy.predict(X_test).numpy()
    acc[rd] = round((1.0*(y_test == y_pred)).sum().item() / len(y_test), 3)
    print('Testing accuracy:', acc[rd], flush=True)
    # if acc[rd] > 0.98:
    #     print('Testing accuracy reached above 98%, stopping training!')
    #     break
print('Training Completed')