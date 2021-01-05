import torch.nn.functional as F
from torch import nn
import numpy as np

class mlpMod(nn.Module):
    def __init__(self, dim, nClasses, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, nClasses)

    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embSize