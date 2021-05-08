import numpy as np
import sys
sys.path.append('../')
from distil.active_learning_strategies.glister import GLISTER
from distil.active_learning_strategies.badge import BADGE
from distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.active_learning_strategies.random_sampling import RandomSampling
from distil.active_learning_strategies.least_confidence import LeastConfidence
from distil.active_learning_strategies.margin_sampling import MarginSampling
from distil.active_learning_strategies.core_set import CoreSet
from distil.active_learning_strategies.fass import FASS

from distil.utils.models.cifar10net import CifarNet
from distil.utils.data_handler import DataHandler_CIFAR10
from distil.utils.dataset import get_dataset
from distil.utils.train_helper import data_train


data_set_name = 'CIFAR10'
download_path = '../downloaded_data/'
X, y, X_test, y_test = get_dataset(data_set_name, download_path)
dim = np.shape(X)[1:]
handler = DataHandler_CIFAR10

X_tr = X[:2000]
y_tr = y[:2000]
X_unlabeled = X[2000:]
y_unlabeled = y[2000:]

X_test = X_test
y_test = y_test.numpy()

nclasses = 10
n_rounds = 11    ##Number of rounds to run active learning
budget = 1000 
print('Nclasses ', nclasses)

# net = ResNet18(channel=1)
# net = mlpMod(dim, nclasses, embSize=24)
net = CifarNet()
train_args = {'n_epoch':250, 'lr':float(0.01), 'batch_size':20} 
strategy_args = {'batch_size' : 20}
strategy = BADGE(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)

# strategy_args = {'batch_size' : 1, 'submod' : 'facility_location', 'selection_type' : 'PerClass'} 
# strategy = FASS(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)

# strategy_args = {'batch_size' : 20}
# strategy = EntropySampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses)
# strategy = RandomSampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
# strategy = LeastConfidence(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = MarginSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses)

# strategy_args = {'batch_size' : 20, 'n_drop' : 2}
# strategy = EntropySamplingDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
# strategy = LeastConfidenceDropout(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
# strategy = MarginSamplingDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)

# strategy_args = {'batch_size' : 20, 'tor':1e-4}
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
    acc[rd] = round((1.0*(y_test == y_pred)).sum().item() / len(y_test), 3)
    print('Testing accuracy:', acc[rd], flush=True)
    # if acc[rd] > 0.98:
    #     print('Testing accuracy reached above 98%, stopping training!')
    #     break
print('Training Completed')