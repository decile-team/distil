from distil.scalable_active_learning_strategies.score_streaming_strategy import ScoreStreamingStrategy

class MarginSampling(ScoreStreamingStrategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(MarginSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob(unlabeled_buffer)
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:,1] - probs_sorted[:, 0] # Margin negated => Largest score corresponds to smallest margin
        return U
