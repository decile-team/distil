from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.optim as optim
import sys
sys.path.append('../')  

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
        
        if 'isverbose' not in args:
            self.args['isverbose'] = False
        
        if 'isreset' not in args:
            self.args['isreset'] = True

        if 'max_accuracy' not in args:
            self.args['max_accuracy'] = 0.95
            
        if 'criterion' not in args:
            self.args['criterion'] = nn.CrossEntropyLoss()

    def update_index(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def update_data(self, X, Y):
    	self.X = X
    	self.Y = Y

    def get_acc_on_set(self, X_test, Y_test):
        
        if X_test is None:
            raise ValueError("Test data not present")
        
        if Y_test is None:
            raise ValueError("Test labels not present")
            
        if X_test.shape[0] != Y_test.shape[0]:
            raise ValueError("X_test has {self.X_test.shape[0]} values but {self.Y_test.shape[0]} labels")
        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1 
        
        loader_te = DataLoader(self.handler(X_test, Y_test, False, use_test_transform=True), shuffle=False, pin_memory=True, batch_size=batch_size)
        self.clf.eval()
        accFinal = 0.

        with torch.no_grad():
            for batch_id, (x,y,idxs) in enumerate(loader_te):
                if self.use_cuda:
                    self.clf = self.clf.cuda()
                    x, y = x.cuda(), y.cuda()
                out = self.clf(x)
                accFinal += torch.sum(1.0*(torch.max(out,1)[1] == y)).item() #.data.item()

        return accFinal / len(loader_te.dataset.X)

    def _train_weighted(self, epoch, loader_tr, optimizer, gradient_weights):
        self.clf.train()
        accFinal = 0.
        criterion = self.args['criterion']
        criterion.reduction = "none"

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
                gradient_weights = gradient_weights.cuda()

            optimizer.zero_grad()
            out = self.clf(x)

            # Modify the loss function to apply weights before reducing to a mean
            loss = criterion(out, y.long())

            # Perform a dot product with the loss vector and the weight vector, then divide by batch size.
            weighted_loss = torch.dot(loss, gradient_weights[idxs])
            weighted_loss = torch.div(weighted_loss, len(idxs))

            accFinal += torch.sum(torch.eq(torch.max(out,1)[1],y)).item() #.data.item()

            # Backward now does so on the weighted loss, not the regular mean loss
            weighted_loss.backward() 

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset.X), weighted_loss

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        criterion = self.args['criterion']

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = self.clf(x)
            loss = criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset.X), loss

    
    def train(self, gradient_weights=None):

        print('Training..')
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        train_logs = []
        n_epoch = self.args['n_epoch']
        
        if self.args['isreset']:
            if self.use_cuda:
                self.clf =  self.net.apply(weight_reset).cuda()
            else:
                self.clf =  self.net.apply(weight_reset)
        else:
            try:
                self.clf
            except:
                if self.use_cuda:
                    self.clf =  self.net.apply(weight_reset).cuda()
                else:
                    self.clf =  self.net.apply(weight_reset)

        optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
        lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.handler(self.X, self.Y, False), batch_size=batch_size, shuffle=True, pin_memory=True)
        epoch = 1
        accCurrent = 0
        while accCurrent < self.args['max_accuracy'] and epoch < n_epoch: 
            
            if gradient_weights is None:
                accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            else:
                accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights)
            lr_sched.step()
            
            epoch += 1
            if(self.args['isverbose']):
                print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

            log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
            train_logs.append(log_string)
            if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        print('Epoch:', str(epoch), 'Training accuracy:', round(accCurrent, 3), flush=True)
        if self.args['islogs']:
            return self.clf, train_logs
        else:
            return self.clf