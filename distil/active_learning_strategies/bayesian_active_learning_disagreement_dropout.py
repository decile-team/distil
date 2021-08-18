import torch

from .score_streaming_strategy import ScoreStreamingStrategy

class BALDDropout(ScoreStreamingStrategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(BALDDropout, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob_dropout_split(unlabeled_buffer, self.n_drop)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        
        # Previous strategy sorts and takes smallest (entropy2 - entropy1). 
        # This one will take largest (entropy1 - entropy2) => smallest (entropy2-entropy1)
        U = entropy1 - entropy2 
        return U