import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def get_dataset(name, path):
    
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te
    return X_tr, Y_tr, X_te, Y_te

# def get_handler(name):
#     if name == 'MNIST':
#         return DataHandler3
#     elif name == 'FashionMNIST':
#         return DataHandler1
#     elif name == 'SVHN':
#         return DataHandler2
#     elif name == 'CIFAR10':
#         return DataHandler3
#     else:
#         return DataHandler4


# class DataHandler3(Dataset):
#     def __init__(self, X, Y, transform=None):
#         self.X = X
#         self.Y = Y
#         self.transform = transform

#     def __getitem__(self, index):
#         x, y = self.X[index], self.Y[index]
#         if self.transform is not None:
#             x = Image.fromarray(x)
#             x = self.transform(x)
#         return x, y, index

#     def __len__(self):
#         return len(self.X)
