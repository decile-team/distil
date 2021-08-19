#import pandas as pd 
import numpy as np
from torch import nn
import torch
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append('../')
from distil.utils.utils import LabeledToUnlabeledDataset
from distil.active_learning_strategies.glister import GLISTER
from distil.utils.models.simple_net import TwoLayerNet
from distil.utils.train_helper import data_train

from torch.utils.data import TensorDataset

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

dset_name = 'satimage'

#User Execution
if dset_name == "satimage":
    trn_file = '../../datasets/satimage/satimage.scale.trn'
    val_file = '../../datasets/satimage/satimage.scale.val'
    tst_file = '../../datasets/satimage/satimage.scale.tst'
    data_dims = 36
    num_cls = 6

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

elif dset_name == "ijcnn1":
    
    trn_file = '../../datasets/ijcnn1/ijcnn1.trn'
    val_file = '../../datasets/ijcnn1/ijcnn1.val'
    tst_file = '../../datasets/ijcnn1/ijcnn1.tst'
    data_dims = 22
    num_cls = 2
    x_trn, y_trn = libsvm_file_load(trn_file, dim=data_dims)
    x_val, y_val = libsvm_file_load(val_file, dim=data_dims)
    x_tst, y_tst = libsvm_file_load(tst_file, dim=data_dims)
    
    # The class labels are (-1,1). Make them to (0,1)
    y_trn[y_trn < 0] = 0
    y_val[y_val < 0] = 0
    y_tst[y_tst < 0] = 0    

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

net = TwoLayerNet(dim, num_cls,100)
net.apply(init_weights)

training_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
unlabeled_dataset = TensorDataset(torch.tensor(X_unlabeled), torch.tensor(y_unlabeled, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(x_tst), torch.tensor(y_tst, dtype=torch.long))

strategy_args = {'batch_size' : 100, 'lr':float(0.001)} 
strategy = GLISTER(training_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, num_cls, strategy_args, validation_dataset = None, typeOf='Diversity', lam=10)

train_args = {'n_epoch':150, 'lr':float(0.001)}  #Different args than strategy_args
n_rounds = 10    ##Number of rounds to run ac
budget = 32    ##Number of new data points after every iteration

#Training first set of points
dt = data_train(training_dataset, net, train_args)
clf = dt.train()
strategy.update_model(clf)
y_pred = strategy.predict(LabeledToUnlabeledDataset(test_dataset)).cpu().numpy()

acc = np.zeros(n_rounds)
acc[0] = (1.0*(y_tst == y_pred)).sum().item() / len(y_tst)
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

    training_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    unlabeled_dataset = TensorDataset(torch.tensor(X_unlabeled), torch.tensor(y_unlabeled, dtype=torch.long))

    #Reload state and start training
    strategy.update_data(training_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
    dt.update_data(training_dataset)

    clf = dt.train()
    strategy.update_model(clf)
    y_pred = strategy.predict(LabeledToUnlabeledDataset(test_dataset)).cpu().numpy()
    acc[rd] = round(1.0 * (y_tst == y_pred).sum().item() / len(y_tst), 3)
    print('Testing accuracy:', acc[rd], flush=True)
    if acc[rd] > 0.98:
        print('Testing accuracy reached above 98%, stopping training!')
        break
print('Training Completed')
# final_df = pd.DataFrame(X_tr)
# final_df['Target'] = list(y_tr)
# final_df.to_csv('../final.csv', index=False)