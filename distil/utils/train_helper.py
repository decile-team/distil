from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import torch.optim as optim
import sys
sys.path.append('../')  

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class AddIndexDataset(Dataset):
    
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset
        
    def __getitem__(self, index):
        data, label = self.wrapped_dataset[index]
        return data, label, index
    
    def __len__(self):
        return len(self.wrapped_dataset)

#custom training
class data_train:

    """
    Provides a configurable training loop for AL.
    
    Parameters
    ----------
    training_dataset: torch.utils.data.Dataset
        The training dataset to use
    net: torch.nn.Module
        The model to train
    args: dict
        Additional arguments to control the training loop
        
        `batch_size` - The size of each training batch (int, optional)
        `islogs`- Whether to return training metadata (bool, optional)
        `optimizer`- The choice of optimizer. Must be one of 'sgd' or 'adam' (string, optional)
        `isverbose`- Whether to print more messages about the training (bool, optional)
        `isreset`- Whether to reset the model before training (bool, optional)
        `max_accuracy`- The training accuracy cutoff by which to stop training (float, optional)
        `min_diff_acc`- The minimum difference in accuracy to measure in the window of monitored accuracies. If all differences are less than the minimum, stop training (float, optional)
        `window_size`- The size of the window for monitoring accuracies. If all differences are less than 'min_diff_acc', then stop training (int, optional)
        `criterion`- The criterion to use for training (typing.Callable[], optional)
        `device`- The device to use for training (string, optional)
    """
    
    def __init__(self, training_dataset, net, args):

        self.training_dataset = AddIndexDataset(training_dataset)
        self.net = net
        self.args = args
        
        self.n_pool = len(training_dataset)
        
        if 'islogs' not in args:
            self.args['islogs'] = False

        if 'optimizer' not in args:
            self.args['optimizer'] = 'sgd'
        
        if 'isverbose' not in args:
            self.args['isverbose'] = False
        
        if 'isreset' not in args:
            self.args['isreset'] = True

        if 'max_accuracy' not in args:
            self.args['max_accuracy'] = 0.95

        if 'min_diff_acc' not in args: #Threshold to monitor for
            self.args['min_diff_acc'] = 0.001

        if 'window_size' not in args:  #Window for monitoring accuracies
            self.args['window_size'] = 10
            
        if 'criterion' not in args:
            self.args['criterion'] = nn.CrossEntropyLoss()
            
        if 'device' not in args:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = args['device']

    def update_index(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def update_data(self, new_training_dataset):
        """
        Updates the training dataset with the provided new training dataset
        
        Parameters
        ----------
        new_training_dataset: torch.utils.data.Dataset
            The new training dataset
        """
        self.training_dataset = AddIndexDataset(new_training_dataset)

    def get_acc_on_set(self, test_dataset):
        
        """
        Calculates and returns the accuracy on the given dataset to test
        
        Parameters
        ----------
        test_dataset: torch.utils.data.Dataset
            The dataset to test
        Returns
        -------
        accFinal: float
            The fraction of data points whose predictions by the current model match their targets
        """	
        
        try:
            self.clf
        except:
            self.clf = self.net

        if test_dataset is None:
            raise ValueError("Test data not present")
        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1 
        
        loader_te = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=batch_size)
        self.clf.eval()
        accFinal = 0.

        with torch.no_grad():        
            self.clf = self.clf.to(device=self.device)
            for batch_id, (x,y) in enumerate(loader_te):     
                x, y = x.to(device=self.device), y.to(device=self.device)
                out = self.clf(x)
                accFinal += torch.sum(1.0*(torch.max(out,1)[1] == y)).item() #.data.item()

        return accFinal / len(test_dataset)

    def _train_weighted(self, epoch, loader_tr, optimizer, gradient_weights):
        self.clf.train()
        accFinal = 0.
        criterion = self.args['criterion']
        criterion.reduction = "none"

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(device=self.device), y.to(device=self.device)
            gradient_weights = gradient_weights.to(device=self.device)

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
        return accFinal / len(loader_tr.dataset), weighted_loss

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        criterion = self.args['criterion']
        criterion.reduction = "mean"

        for batch_id, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(device=self.device), y.to(device=self.device)

            optimizer.zero_grad()
            out = self.clf(x)
            loss = criterion(out, y.long())
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).item()
            loss.backward()

            # clamp gradients, just in case
            # for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
        return accFinal / len(loader_tr.dataset), loss

    def check_saturation(self, acc_monitor):
        
        saturate = True

        for i in range(len(acc_monitor)):
            for j in range(i+1, len(acc_monitor)):
                if acc_monitor[j] - acc_monitor[i] >= self.args['min_diff_acc']:
                    saturate = False
                    break

        return saturate

    def train(self, gradient_weights=None):

        """
        Initiates the training loop.
        
        Parameters
        ----------
        gradient_weights: list, optional
            The weight of each data point's effect on the loss gradient. If none, regular training will commence. If not, weighted training will commence.
        Returns
        -------
        model: torch.nn.Module
            The trained model. Alternatively, this will also return the training logs if 'islogs' is set to true.
        """        

        print('Training..')
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        train_logs = []
        n_epoch = self.args['n_epoch']
        
        if self.args['isreset']:
            self.clf = self.net.apply(weight_reset).to(device=self.device)
        else:
            try:
                self.clf
            except:
                self.clf = self.net.apply(weight_reset).to(device=self.device)

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
            lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
        
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        
        if 'batch_size' in self.args:
            batch_size = self.args['batch_size']
        else:
            batch_size = 1

        # Set shuffle to true to encourage stochastic behavior for SGD
        loader_tr = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        epoch = 1
        accCurrent = 0
        is_saturated = False
        acc_monitor = []

        while (accCurrent < self.args['max_accuracy']) and (epoch < n_epoch) and (not is_saturated): 
            
            if gradient_weights is None:
                accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            else:
                accCurrent, lossCurrent = self._train_weighted(epoch, loader_tr, optimizer, gradient_weights)
            
            acc_monitor.append(accCurrent)

            if self.args['optimizer'] == 'sgd':
                lr_sched.step()
            
            epoch += 1
            if(self.args['isverbose']):
                if epoch % 50 == 0:
                    print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

            #Stop training if not converging
            if len(acc_monitor) >= self.args['window_size']:

                is_saturated = self.check_saturation(acc_monitor)
                del acc_monitor[0]

            log_string = 'Epoch:' + str(epoch) + '- training accuracy:'+str(accCurrent)+'- training loss:'+str(lossCurrent)
            train_logs.append(log_string)
            if (epoch % 50 == 0) and (accCurrent < 0.2): # resetif not converging
                self.clf = self.net.apply(weight_reset).to(device=self.device)
                
                if self.args['optimizer'] == 'sgd':

                    optimizer = optim.SGD(self.clf.parameters(), lr = self.args['lr'], momentum=0.9, weight_decay=5e-4)
                    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

                else:
                    optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        print('Epoch:', str(epoch), 'Training accuracy:', round(accCurrent, 3), flush=True)

        if self.args['islogs']:
            return self.clf, train_logs
        else:
            return self.clf