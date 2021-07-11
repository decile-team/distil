from distil.scalable_active_learning_strategies.score_streaming_strategy import ScoreStreamingStrategy

class LeastConfidenceDropout(ScoreStreamingStrategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(LeastConfidenceDropout, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    
        if 'n_drop' in args:
            self.n_drop = args['n_drop']
        else:
            self.n_drop = 10
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob_dropout(unlabeled_buffer, self.n_drop)
        U = -probs.max(1)[0] # Max prob. negated => Largest score corresponds to smallest max prob (least confident prediction)
        return U