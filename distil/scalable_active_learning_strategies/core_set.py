import torch

from torch.utils.data import Dataset
from .strategy import Strategy

class CoreSet(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(CoreSet, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
  
    def furthest_first(self, unlabeled_embeddings, labeled_embeddings, n):
        
        unlabeled_embeddings = unlabeled_embeddings.to(self.device)
        labeled_embeddings = labeled_embeddings.to(self.device)
        
        m = unlabeled_embeddings.shape[0]
        if labeled_embeddings.shape[0] == 0:
            min_dist = torch.tile(float("inf"), m)
        else:
            dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
            min_dist = torch.min(dist_ctr, dim=1)[0]
  
        idxs = []
        
        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx)
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j,0])
                
        return idxs
  
    def select(self, budget):
        
        class NoLabelDataset(Dataset):
            
            def __init__(self, wrapped_dataset):
                self.wrapped_dataset = wrapped_dataset
                
            def __getitem__(self, index):
                features, label = self.wrapped_dataset[index]
                return features
            
            def __len__(self):
                return len(self.wrapped_dataset)
        
        embedding_unlabeled = self.get_embedding(self.unlabeled_dataset)
        embedding_labeled = self.get_embedding(NoLabelDataset(self.labeled_dataset))
        chosen = self.furthest_first(embedding_unlabeled, embedding_labeled, budget)

        return chosen        