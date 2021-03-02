#import pandas as pd 
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
from sklearn.preprocessing import StandardScaler
import argparse
sys.path.append('../../')
from distil.utils.DataHandler import DataHandler_Points
from distil.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, \
                            LeastConfidence, MarginSampling, CoreSet, AdversarialBIM, AdversarialDeepFool, \
                            KMeansSampling, BaselineSampling, BALDDropout
from distil.utils.models.simpleNN_net import TwoLayerNet
from distil.utils.TrainHelper import data_train

def test_strategy(selected_strat):
    """
    Function for testing individual active learning strategy

    Parameters
    ----------
    selected_strat: String
        Strategy to be tested.
        badge, glister, entropy_sampling, random_sampling
        margin_sampling, least_confidence, core_set, bald_dropout, adversarial_bim,
        kmeans_sampling, baseline_sampling, adversarial_deepfool
    """

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def libsvm_file_load(path,dim, save_data=False):
        data = []
        target = []
        with open(path) as fp:
           line = fp.readline()
           while line:
            temp = [i for i in line.strip().split(" ")]
            target.append(int(float(temp[0]))) # Class Number. # Not assumed to be in (0, K-1)
            temp_data = [0]*dim
            
            for i in temp[1:]:
                ind,val = i.split(':')
                temp_data[int(ind)-1] = float(val)
            data.append(temp_data)
            line = fp.readline()
        X_data = np.array(data,dtype=np.float32)
        Y_label = np.array(target)
        if save_data:
            # Save the numpy files to the folder where they come from
            data_np_path = path + '.data.npy'
            target_np_path = path + '.label.npy'
            np.save(data_np_path, X_data)
            np.save(target_np_path, Y_label)
        return (X_data, Y_label)

    print('Strategy to be tested on -', selected_strat)

    dset_name = 'satimage'

    #User Execution
    trn_file = '../../datasets/satimage/satimage.scale.trn'
    val_file = '../../datasets/satimage/satimage.scale.val'
    tst_file = '../../datasets/satimage/satimage.scale.tst'
    data_dims = 36
    nclasses = 6

    x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
    x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
    x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)

    y_trn -= 1  # First Class should be zero
    y_val -= 1
    y_tst -= 1  # First Class should be zero

    sc = StandardScaler()
    x_trn = sc.fit_transform(x_trn)
    x_val = sc.transform(x_val)
    x_tst = sc.transform(x_tst)


    nSamps, dim = np.shape(x_trn)

    np.random.seed(42)
    start_idxs = np.random.choice(nSamps, size=32, replace=False)

    X_tr = x_trn[start_idxs]
    X_unlabeled = np.delete(x_trn, start_idxs, axis = 0)

    y_tr = y_trn[start_idxs]
    y_unlabeled = np.delete(y_trn, start_idxs, axis = 0)

    net = TwoLayerNet(dim, nclasses,100)
    net.apply(init_weights)

    strategy_args = {'batch_size' : 100, 'lr':float(0.001)} 
    if selected_strat == 'badge':
        strategy = BADGE(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    elif selected_strat == 'glister':
        strategy = GLISTER(X_tr, y_tr, X_unlabeled, net, DataHandler_Points,nclasses, strategy_args,valid=False,\
                    typeOf='Diversity',lam=10)
    elif selected_strat == 'entropy_sampling':
        strategy = EntropySampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses)
    elif selected_strat == 'margin_sampling':
        strategy = MarginSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses)
    elif selected_strat == 'least_confidence':
        strategy = LeastConfidence(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    elif selected_strat == 'core_set':
        strategy = CoreSet(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    elif selected_strat == 'random_sampling':
        strategy = RandomSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    elif selected_strat == 'bald_dropout':
        strategy = BALDDropout(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    elif selected_strat == 'adversarial_bim':
        strategy = AdversarialBIM(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    elif selected_strat == 'kmeans_sampling':
        strategy = KMeansSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    elif selected_strat == 'baseline_sampling':
        strategy = BaselineSampling(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    elif selected_strat == 'adversarial_deepfool':
        strategy = AdversarialDeepFool(X_tr, y_tr, X_unlabeled, net, DataHandler_Points, nclasses, strategy_args)
    else:
        print('Enter a valid strategy, for more info: python TestStrategy.py -h')
        sys.exit()

    train_args = {'n_epoch':150, 'lr':float(0.001)}  #Different args than strategy_args
    n_rounds = 3    ##Number of rounds to run ac
    budget = 32    ##Number of new data points after every iteration

    #Training first set of points
    dt = data_train(X_tr, y_tr, net, DataHandler_Points, train_args)
    clf = dt.train()
    strategy.update_model(clf)
    y_pred = strategy.predict(x_tst).numpy()

    acc = np.zeros(n_rounds)
    acc[0] = (1.0*(y_tst == y_pred)).sum().item() / len(y_tst)

    print('***************************')
    print('Starting Strategy Testing..')
    print('***************************')
    ##User Controlled Loop
    try:
        for rd in range(1, n_rounds):

            idx = strategy.select(budget)
            strategy.save_state()

            #Adding new points to training set
            X_tr = np.concatenate((X_tr, X_unlabeled[idx]), axis=0)
            X_unlabeled = np.delete(X_unlabeled, idx, axis = 0)

            #Human In Loop, Assuming user adds new labels here
            y_tr = np.concatenate((y_tr, y_unlabeled[idx]), axis = 0)
            y_unlabeled = np.delete(y_unlabeled, idx, axis = 0)

            #Reload state and start training
            strategy.load_state()
            strategy.update_data(X_tr, y_tr, X_unlabeled)
            dt.update_data(X_tr, y_tr)

            clf = dt.train()
            strategy.update_model(clf)
            y_pred = strategy.predict(x_tst).numpy()
            acc[rd] = round(1.0 * (y_tst == y_pred).sum().item() / len(y_tst), 3)

        print('Test Successful for strategy -', selected_strat)
    except:
        print('Test unsuccessful for strategy -', selected_strat)

# test_strategy('badge')