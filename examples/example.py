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

sys.path.append('../')
from utils.DataHandler import DataHandler_Points
from active_learning_strategies import GLISTER, BADGE


# linear model class
class linMod(nn.Module):
    def __init__(self, nc=1, sz=28):
        super(linMod, self).__init__()
        self.lm = nn.Linear(int(np.prod(dim)), opts.nClasses)
    def forward(self, x):
        x = x.view(-1, int(np.prod(dim)))
        out = self.lm(x)
        return out, x
    def get_embedding_dim(self):
        return int(np.prod(dim))

class mlpMod(nn.Module):
    def __init__(self, dim, nClasses, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, nClasses)

    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embSize

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
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset.X)

    
    def train(self):

        print('Training..')
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args['n_epoch']
        if self.use_cuda:
            self.clf =  self.net.apply(weight_reset).cuda()
        else:
            self.clf =  self.net.apply(weight_reset)

        optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
        loader_tr = DataLoader(self.handler(self.X, self.Y, False))
        epoch = 1
        accCurrent = 0
        while accCurrent < 0.95 and epoch < n_epoch: 
            accCurrent = self._train(epoch, loader_tr, optimizer)
            epoch += 1
            # print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
            
            if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        print('Epoch:', str(epoch),'Training accuracy:',round(accCurrent, 3), flush=True)
        return self.clf

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

#User Execution
trn_file = '../datasets/satimage/satimage.scale.trn'
val_file = '../datasets/satimage/satimage.scale.val'
tst_file = '../datasets/satimage/satimage.scale.tst'
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

nSamps, dim = np.shape(x_trn)

np.random.seed(42)
start_idxs = np.random.choice(nSamps, size=32, replace=False)

X_tr = x_trn[start_idxs]
X_unlabeled = np.delete(x_trn, start_idxs, axis = 0)

y_tr = y_trn[start_idxs]
y_unlabeled = np.delete(y_trn, start_idxs, axis = 0)

net = mlpMod(dim, num_cls, embSize=100)
net.apply(init_weights)

strategy_args = {'batch_size' : 100, 'lr':float(0.001)} 
strategy = GLISTER(X_tr, y_tr, X_unlabeled, net, DataHandler_Points,num_cls, strategy_args,valid=False,
typeOf='Diversity',lam=10)
#,X_val=x_val,Y_val=y_val)

#valid,X_val=None,Y_val=None,loss_criterion=nn.CrossEntropyLoss(),typeOf='none',lam=None,\
#    kernel_batch_size = 200

args = {'n_epoch':150, 'lr':float(0.001)}  #Different args than strategy_args
n_rounds = 10    ##Number of rounds to run ac
budget = 32    ##Number of new data points after every iteration

#Training first set of points
dt = data_train(X_tr, y_tr, net, DataHandler_Points, args)
clf = dt.train()
strategy.update_model(clf)
y_pred = strategy.predict(x_tst).numpy()

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
    y_pred = strategy.predict(x_tst).numpy()
    acc[rd] = round(1.0 * (y_tst == y_pred).sum().item() / len(y_tst), 3)
    print('Testing accuracy:', acc[rd], flush=True)
    if acc[rd] > 0.98:
        print('Testing accuracy reached above 98%, stopping training!')
        break
print('Training Completed')
# final_df = pd.DataFrame(X_tr)
# final_df['Target'] = list(y_tr)
# final_df.to_csv('../final.csv', index=False)