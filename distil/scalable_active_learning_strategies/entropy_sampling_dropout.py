import torch

from distil.scalable_active_learning_strategies.score_streaming_strategy import ScoreStreamingStrategy

class EntropySampling(ScoreStreamingStrategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(EntropySampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob_dropout(unlabeled_buffer, self.n_drop)
        log_probs = torch.log(probs)
        U = -(probs*log_probs).sum(1)
        return U