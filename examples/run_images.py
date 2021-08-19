import numpy as np
import sys
sys.path.append('../')
from distil.active_learning_strategies.badge import BADGE

from distil.utils.models.cifar10net import CifarNet
from distil.utils.train_helper import data_train
from distil.utils.utils import LabeledToUnlabeledDataset

from torch.utils.data import Subset, ConcatDataset

from torchvision import transforms
from torchvision.datasets import cifar

data_set_name = 'CIFAR10'
download_path = '../../datasets/downloaded_data/'

cifar_training_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
cifar_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

cifar10_full_train = cifar.CIFAR10(download_path, train=True, transform=cifar_training_transform, download=True)
cifar10_test = cifar.CIFAR10(download_path, train=False, transform=cifar_test_transform, download=True)

y_test = cifar10_test.targets

dim = np.shape(cifar10_full_train[0][0])

train_set_size = 2000

cifar10_train = Subset(cifar10_full_train, list(range(train_set_size)))
cifar10_unlabeled = Subset(cifar10_full_train, list(range(train_set_size, len(cifar10_full_train))))

nclasses = 10
n_rounds = 11    ##Number of rounds to run active learning
budget = 1000 
print('Nclasses ', nclasses)

net = CifarNet()
train_args = {'n_epoch':250, 'lr':float(0.01), 'batch_size':20} 
strategy_args = {'batch_size' : 20}
strategy = BADGE(cifar10_train, LabeledToUnlabeledDataset(cifar10_unlabeled), net, nclasses, strategy_args)

#Training first set of points
dt = data_train(cifar10_train, net, train_args)
clf = dt.train()
strategy.update_model(clf)
y_pred = strategy.predict(LabeledToUnlabeledDataset(cifar10_test)).cpu().numpy()

acc = np.zeros(n_rounds)
acc[0] = (1.0*(y_test == y_pred)).sum().item() / len(y_test)
print('Initial Testing accuracy:', round(acc[0], 3), flush=True)

##User Controlled Loop
for rd in range(1, n_rounds):
    print('-------------------------------------------------')
    print('Round', rd) 
    print('-------------------------------------------------')
    cifar10_full_train.transform = cifar_test_transform # Disable augmentation
    idx = strategy.select(budget)
    cifar10_full_train.transform = cifar_training_transform # Re-enable augmentation
    print('New data points added -', len(idx))

    #Adding new points to training set
    cifar10_train = ConcatDataset([cifar10_train, Subset(cifar10_unlabeled, idx)])
    remaining_unlabeled_idx = list(set(range(len(cifar10_unlabeled))) - set(idx))
    cifar10_unlabeled = Subset(cifar10_unlabeled, remaining_unlabeled_idx)
    
    print('Number of training points -', len(cifar10_train))
    print('Number of labels -', len(cifar10_train))
    print('Number of unlabeled points -', len(cifar10_unlabeled))

    #Reload state and start training
    strategy.update_data(cifar10_train, LabeledToUnlabeledDataset(cifar10_unlabeled))
    dt.update_data(cifar10_train)

    clf = dt.train()
    strategy.update_model(clf)
    y_pred = strategy.predict(LabeledToUnlabeledDataset(cifar10_test)).cpu().numpy()
    acc[rd] = round((1.0*(y_test == y_pred)).sum().item() / len(y_test), 3)
    print('Testing accuracy:', acc[rd], flush=True)
    # if acc[rd] > 0.98:
    #     print('Testing accuracy reached above 98%, stopping training!')
    #     break
print('Training Completed')