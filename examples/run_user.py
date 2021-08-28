import pandas as pd 
import numpy as np
from torch import nn
import torch
import sys
sys.path.append('../')
from distil.active_learning_strategies.core_set import CoreSet   

from distil.utils.models.simple_net import TwoLayerNet
from distil.utils.train_helper import data_train
from distil.utils.utils import LabeledToUnlabeledDataset

from torch.utils.data import TensorDataset

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
budget = 2 		##Number of new data points after every iteration
strategy_args = {'batch_size' : 2, 'lr' : 0.1} 

df = pd.read_csv(data_path)
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:,:-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

X_tr = X[:10]    #First set of labeled examples
y_tr = y[:10]

X_unlabeled = X[10:]		#Unlabeled data
y_unlabeled = y[10:]			

df_test = pd.read_csv(test_path)
X_test = df_test.iloc[:,:-1].to_numpy()
y_test = df_test.iloc[:, -1].to_numpy()

nSamps, dim = np.shape(X)

net = TwoLayerNet(dim, nclasses, dim*2)
net.apply(init_weights)

training_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float), torch.tensor(y_tr, dtype=torch.long))
unlabeled_dataset = TensorDataset(torch.tensor(X_unlabeled, dtype=torch.float), torch.tensor(y_unlabeled, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))

strategy_args = {'batch_size' : 10, 'tor':1e-4}
strategy = CoreSet(training_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, nclasses, strategy_args)

#Training first set of points
dt = data_train(training_dataset, net, train_args)
clf = dt.train()
strategy.update_model(clf)
y_pred = strategy.predict(LabeledToUnlabeledDataset(test_dataset)).cpu().numpy()

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
    
    #Adding new points to training set
    X_tr = np.concatenate((X_tr, X_unlabeled[idx]), axis=0)
    X_unlabeled = np.delete(X_unlabeled, idx, axis = 0)

    #Human In Loop, Assuming user adds new labels here
    y_tr = np.concatenate((y_tr, y_unlabeled[idx]), axis = 0)
    y_unlabeled = np.delete(y_unlabeled, idx, axis = 0)
    print('Number of training points -',X_tr.shape[0])
    print('Number of labels -', y_tr.shape[0])
    print('Number of unlabeled points -', X_unlabeled.shape[0])

    training_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float), torch.tensor(y_tr, dtype=torch.long))
    unlabeled_dataset = TensorDataset(torch.tensor(X_unlabeled, dtype=torch.float), torch.tensor(y_unlabeled, dtype=torch.long))

    strategy.update_data(training_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
    dt.update_data(training_dataset)

    clf = dt.train()
    strategy.update_model(clf)
    y_pred = strategy.predict(LabeledToUnlabeledDataset(test_dataset)).cpu().numpy()
    acc[rd] = round(1.0 * (y_test == y_pred).sum().item() / len(y_test), 3)
    print('Testing accuracy:', acc[rd], flush=True)
    # if acc[rd] > 0.98:
    #     print('Testing accuracy reached above 98%, stopping training!')
    #     break
print('Training Completed')