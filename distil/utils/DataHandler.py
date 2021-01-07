from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms

class DataHandler_Points(Dataset):
    """
    Data Handler to load data points.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """
    def __init__(self, X, Y=None, select=True):
        """
        Constructor
        """
        
        self.select = select
        if not self.select:
        	self.X = X.astype(np.float32)
        	self.Y = Y
        else:
        	self.X = X.astype(np.float32)  #For unlabeled Data

    def __getitem__(self, index):
    	if not self.select:
    		x, y = self.X[index], self.Y[index]
    		return x, y, index
    	else:
        	x = self.X[index]              #For unlabeled Data
        	return x, index

    def __len__(self):
        return len(self.X)

class DataHandler_MNIST(Dataset):
    """
    Data Handler to load MNIST dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True):
        """
        Constructor
        """
        self.select = select
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if not self.select:
            self.X = X
            self.Y = Y
            self.transform = transform
        else:
            self.X = X
            self.transform = transform

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x, y, index

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x, index

    def __len__(self):
        return len(self.X)

class DataHandler_CIFAR10(Dataset):
    """
    Data Handler to load CIFAR10 dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True):
        """
        Constructor
        """
        self.select = select
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        if not self.select:
            self.X = X
            self.Y = Y
            self.transform = transform
        else:
            self.X = X
            self.transform = transform

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x, y, index

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x, index

    def __len__(self):
        return len(self.X)