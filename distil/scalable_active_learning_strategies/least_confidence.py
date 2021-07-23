from .score_streaming_strategy import ScoreStreamingStrategy

class LeastConfidence(ScoreStreamingStrategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(LeastConfidence, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
    
    def acquire_scores(self, unlabeled_buffer):
        
        probs = self.predict_prob(unlabeled_buffer)
        U = -probs.max(1)[0] # Max prob. negated => Largest score corresponds to smallest max prob (least confident prediction)
        return U