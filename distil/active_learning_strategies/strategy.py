import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle

class Strategy:
    def __init__(self,X, Y, unlabeled_x, net, handler, nclasses, args={}): #
        
        self.X = X
        self.Y = Y
        self.unlabeled_x = unlabeled_x
        self.model = net
        self.handler = handler
        self.target_classes = nclasses
        self.args = args
        if 'batch_size' not in args:
            args['batch_size'] = 1
        
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']

    def select(self, budget):
        pass

    def update_data(self,X,Y,unlabeled_x): #
        self.X = X
        self.Y = Y
        self.unlabeled_x = unlabeled_x

    def update_model(self, clf):
        self.model = clf

    def save_state(self, filename):


        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_state(self, filename):

        with open(filename, 'rb') as f:
            self = pickle.load(f)

    def predict(self,X, useloader=True):
    
        self.model.eval()
        P = torch.zeros(X.shape[0]).long()
        with torch.no_grad():

            if useloader:
                loader_te = DataLoader(self.handler(X),shuffle=False, batch_size = self.args['batch_size'])
                for x, idxs in loader_te:
                    x = x.to(self.device)  
                    out = self.model(x)
                    pred = out.max(1)[1]
                    P[idxs] = pred.data.cpu()
            else:            
                x=X
                out = self.model(x)
                pred = out.max(1)[1]
                P = pred.data.cpu()

        return P

    def predict_prob(self,X):

        loader_te = DataLoader(self.handler(X),shuffle=False, batch_size = self.args['batch_size'])
        self.model.eval()
        probs = torch.zeros([X.shape[0], self.target_classes])
        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)                  
                out = self.model(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def predict_prob_dropout(self,X, n_drop):

        loader_te = DataLoader(self.handler(X),shuffle=False, batch_size = self.args['batch_size'])
        self.model.train()
        probs = torch.zeros([X.shape[0], self.target_classes])
        with torch.no_grad():
            for i in range(n_drop):
                # print('n_drop {}/{}'.format(i+1, n_drop))
                for x, idxs in loader_te:

                    x = x.to(self.device)   
                    out = self.model(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self,X, n_drop):
        
        loader_te = DataLoader(self.handler(X),shuffle=False, batch_size = self.args['batch_size'])
        self.model.train()
        probs = torch.zeros([n_drop, X.shape[0], self.target_classes])
        with torch.no_grad():
            for i in range(n_drop):
                # print('n_drop {}/{}'.format(i+1, n_drop))
                for x, idxs in loader_te:
                    x = x.to(self.device)
                    out = self.model(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self,X):
        
        loader_te = DataLoader(self.handler(X),shuffle=False, batch_size = self.args['batch_size'])
        self.model.eval()
        embedding = torch.zeros([X.shape[0], self.model.get_embedding_dim()])

        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)  
                out, e1 = self.model(x,last=True)
                embedding[idxs] = e1.data.cpu()
        return embedding

    # gradient embedding (assumes cross-entropy loss)
    #calculating hypothesised labels within
    def get_grad_embedding(self,X,Y=None, bias_grad=True):
        
        embDim = self.model.get_embedding_dim()
        
        nLab = self.target_classes

        if bias_grad:
            embedding = torch.zeros([X.shape[0], (embDim+1)*nLab],device=self.device)
        else:
            embedding = torch.zeros([X.shape[0], embDim * nLab],device=self.device)
        
        loader_te = DataLoader(self.handler(X),shuffle=False, batch_size = self.args['batch_size'])

        with torch.no_grad():
            for x, idxs in loader_te:
                x = x.to(self.device)
                out, l1 = self.model(x,last=True)
                data = F.softmax(out, dim=1)

                outputs = torch.zeros(x.shape[0], nLab).to(self.device)
                if Y is None:
                    y_trn = self.predict(x, useloader=False)
                else:
                    y_trn = torch.tensor(Y[idxs])
                y_trn = y_trn.to(self.device).long()
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

        return embedding
