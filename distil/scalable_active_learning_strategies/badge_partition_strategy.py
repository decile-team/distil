from distil.scalable_active_learning_strategies.partition_strategy import PartitionStrategy
from distil.active_learning_strategies.badge import BADGE

class BADGEPartitionStrategy(PartitionStrategy):
    
    def __init__(self, labeled_dataloader, unlabeled_dataloader, net, nclasses, args={}): #
        
        super(PartitionStrategy, self).__init__(labeled_dataloader, unlabeled_dataloader, net, nclasses, BADGE, args)        