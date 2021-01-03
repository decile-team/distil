from torch.utils.data import Dataset
import numpy as np

class DataHandler_Points(Dataset):
    def __init__(self, X, Y=None, select=True):
        
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