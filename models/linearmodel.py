import torch.nn.functional as F
from torch import nn
import numpy as np

class linMod(nn.Module):
    def __init__(self, dim, nClasses):

        super(linMod, self).__init__()
        self.lm = nn.Linear(int(np.prod(dim)), nClasses)
        self.dim = dim
        self.nClasses = nClasses

    def forward(self, x):
        x = x.view(-1, int(np.prod(self.dim)))
        out = self.lm(x)
        return out, x
    def get_embedding_dim(self):
        return int(np.prod(self.dim))