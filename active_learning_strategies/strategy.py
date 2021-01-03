import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import pickle

class Strategy:
    def __init__(self,X, Y, unlabeled_x, net, handler, nclasses, args): #
        
        self.X = X
        self.Y = Y
        # self.idxs_lb = idxs_lb
        self.unlabeled_x = unlabeled_x
        self.clf = net
        self.handler = handler
        self.target_classes = nclasses
        self.args = args
        # self.n_pool = len(Y)
        self.filename = '../data_corpus/state.pkl'
        #print('Use_CUDA ', self.use_cuda)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def select(self, n):
        pass

    def update_data(self,X,Y,unlabeled_x): #
        self.X = X
        self.Y = Y
        self.unlabeled_x = unlabeled_x

    def update_model(self, clf):
        self.clf = clf

    def save_state(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)

    def load_state(self):
        with open(self.filename, 'rb') as f:
            self = pickle.load(f)

    def predict(self,X):

        loader_te = DataLoader(self.handler(X),shuffle=False,**self.args['batch_size'])

        self.clf.eval()
        P = torch.zeros(X.shape[0]).long()
        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)  
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self,X):

        loader_te = DataLoader(self.handler(X),shuffle=False,**self.args['batch_size'])
        self.clf.eval()
        probs = torch.zeros([X.shape[0], self.target_classes])
        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)                  
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def predict_prob_dropout(self,X, n_drop):
        
        loader_te = DataLoader(self.handler(X),shuffle=False,**self.args['batch_size'])
        self.clf.train()
        probs = torch.zeros([X.shape[0], self.target_classes])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, idxs in loader_te:

                    x = x.to(self.device)   
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self,X, n_drop):
        
        loader_te = DataLoader(self.handler(X),shuffle=False,**self.args['batch_size'])
        self.clf.train()
        probs = torch.zeros([n_drop, X.shape[0], self.target_classes])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, idxs in loader_te:
                    x = x.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self,X):
        
        loader_te = DataLoader(self.handler(X),shuffle=False,
             **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([X.shape[0], self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)  
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding

    # gradient embedding (assumes cross-entropy loss)
    #calculating hypothesised labels within
    def get_grad_embedding(self,X, bias_grad=True):
        
        embDim = self.clf.get_embedding_dim()
        
        nLab = self.target_classes
        
        embedding = np.zeros([X.shape[0], embDim * nLab])
        loader_te = DataLoader(self.handler(X),shuffle=False,
             **self.args['loader_te_args'])

        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)
                out, l1 = self.clf(x)
                data = F.softmax(out, dim=1)

                outputs = torch.zeros(x.shape[0], nLab).to(self.device)
                y_trn = self.predict(x)
                outputs.scatter_(1, y_trn.view(-1, 1), 1)
                l0_grads = data - outputs
                l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                l1_grads = l0_expand * l1.repeat(1, nLab)
                
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()
                
                if bias_grad:
                    embedding[idxs] = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    embedding[idxs] = l1_grads

        '''with torch.no_grad():
            for x, idxs in loader_te:
                if self.use_cuda:
                    x = Variable(x.cuda())
                else:
                    x = Variable(x) 

                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(x.shape[0]):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)'''


    # gradient embedding old function
    
    # def get_grad_embedding(self, X, Y):
    #     model = self.clf
    #     embDim = model.get_embedding_dim()
    #     model.eval()
    #     nLab = len(np.unique(Y))
    #     embedding = np.zeros([len(Y), embDim * nLab])
    #     loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
    #                         shuffle=False, **self.args['loader_te_args'])
    #     with torch.no_grad():
    #         for x, y, idxs in loader_te:
    #             # x, y = Variable(x.cuda()), Variable(y.cuda())
    #             if self.use_cuda:
    #                 x, y = Variable(x.cuda()), Variable(y.cuda())
    #             else:
    #                 x, y = Variable(x), Variable(y)  
    #             cout, out = self.clf(x)
    #             out = out.data.cpu().numpy()
    #             batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
    #             maxInds = np.argmax(batchProbs,1)
    #             for j in range(len(y)):
    #                 for c in range(nLab):
    #                     if c == maxInds[j]:
    #                         embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
    #                     else:
    #                         embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
    #         return torch.Tensor(embedding)
