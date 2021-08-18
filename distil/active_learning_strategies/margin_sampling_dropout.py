from .score_streaming_strategy import ScoreStreamingStrategy

class MarginSamplingDropout(ScoreStreamingStrategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(MarginSamplingDropout, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob_dropout(unlabeled_buffer, self.n_drop)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:,1] - probs_sorted[:, 0] # Margin negated => Largest score corresponds to smallest margin
        return U
