import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def get_dataset(name, path):
    """
    Loads dataset

    Parameters
    ----------
    name: str
        Name of the dataset to be loaded. Supports MNIST and CIFAR10
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)

def get_MNIST(path):
    """
    Downloads MNIST dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    """
    Downloads CIFAR10 dataset

    Parameters
    ----------
    path: str
        Path to save the downloaded dataset

    Returns
    ----------
    X_tr: numpy array
        Train set
    Y_tr: torch tensor
        Training Labels
    X_te: numpy array
        Test Set
    Y_te: torch tensor
        Test labels

    """
    
    # Introduce a training transform that provides generalization in training to the test data.
    training_gen_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True, transform=training_gen_transform)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te
    return X_tr, Y_tr, X_te, Y_te