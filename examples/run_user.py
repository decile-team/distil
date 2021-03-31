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
                                CoreSet, GLISTER, BADGE, AdversarialBIM, AdversarialDeepFool, KMeansSampling, BaselineSampling, \
                                  BALDDropout      

from distil.active_learning_strategies import FASS
from distil.utils.models.logreg_net import LogisticRegNet
from distil.utils.models.simpleNN_net import TwoLayerNet
from distil.utils.DataHandler import DataHandler_Points
from distil.utils.TrainHelper import data_train

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

#User Execution
data_path = '../../datasets/iris.csv'
test_path = '../../datasets/iris_test.csv'
train_args = {'n_epoch':150, 'lr':float(0.001), 'batch_size':5, 'optimizer':'sgd'}  #Training args, Different args than strategy_args
nclasses = 3    ##Number of unique classes
n_rounds = 11    ##Number of rounds to run active learning
budget = 10 		##Number of new data points after every iteration
strategy_args = {'batch_size' : 2, 'lr' : 0.1} 

df = pd.read_csv(data_path)
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:,:-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

X_tr = X[:20]    #First set of labeled examples
y_tr = y[:20]

X_unlabeled = X[20:]		#Unlabeled data
y_unlabeled = y[20:]			

df_test = pd.read_csv(test_path)
X_test = df_test.iloc[:,:-1].to_numpy()
y_test = df_test.iloc[:, -1].to_numpy()

nSamps, dim = np.shape(X)

net = TwoLayerNet(dim, nclasses, dim*2)
net.apply(init_weights)

strategy_args = {'batch_size' : 2, 'submod' : 'feature_based', 'selection_type' : 'Full'} 
strategy = FASS(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

# strategy = BADGE(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

# strategy_args = {'batch_size' : 2}
# strategy = EntropySampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses)
# strategy = RandomSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = LeastConfidence(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = MarginSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses)

# strategy_args = {'batch_size' : 2, 'n_drop' : 2}
# strategy = EntropySamplingDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = LeastConfidenceDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = MarginSamplingDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = BALDDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

# strategy_args = {'batch_size' : 1, 'tor':1e-4}
# strategy = CoreSet(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

# strategy_args = {'batch_size' : 5}
# strategy = AdversarialBIM(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = KMeansSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = BaselineSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy_args = {'batch_size' : 5, 'maxiter': 25}
# strategy = AdversarialDeepFool(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)


#Training first set of points
dt = data_train(X_tr, y_tr, net, DataHandler_Points, train_args)
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
    strategy.save_state('./state.pkl')

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
    strategy.load_state('./state.pkl')
    strategy.update_data(X_tr, y_tr, X_unlabeled)
    dt.update_data(X_tr, y_tr)

    clf = dt.train()
    strategy.update_model(clf)
    y_pred = strategy.predict(X_test).numpy()
    acc[rd] = round(1.0 * (y_test == y_pred).sum().item() / len(y_test), 3)
    print('Testing accuracy:', acc[rd], flush=True)
    # if acc[rd] > 0.98:
    #     print('Testing accuracy reached above 98%, stopping training!')
    #     break
print('Training Completed')