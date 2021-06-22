import math
import numpy as np

from torch.utils.data import Subset
from distil.scalable_active_learning_strategies.strategy import Strategy

class PartitionStrategy(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}): #
        
        super(PartitionStrategy, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if "num_partitions" not in args:
            self.num_partitions = 1
        else:
            self.num_partitions = args["num_partitions"]
            
        if "wrapped_strategy_class" not in args:
            raise ValueError("args dictionary requires 'wrapped_strategy_class' key")
            
        self.wrapped_strategy_class = args["wrapped_strategy_class"]
        
    def select(self, budget):
        
        # The number of partitions should be less than or equal to the budget.
        # This is because the budget is evenly divided among the partitions (roughly),
        # so having a smaller budget than the number of partitions results in one or 
        # more partitions having a 0 budget, which should not happen.
        if self.num_partitions > budget:
            raise ValueError("Budget cannot be less than the number of partitions!")
        
        # Furthermore, the number of partitions cannot be more than the size of the unlabeled set
        if self.num_partitions > len(self.unlabeled_dataset):
            raise ValueError("There cannot be more partitions than the size of the dataset!")
    
        # Calculate partition splits and budgets for each partition
        full_unlabeled_size = len(self.unlabeled_dataset)
        split_indices = [math.ceil(full_unlabeled_size * ((1+x) / self.num_partitions)) for x in range(self.num_partitions)]        
        partition_budget_splits = [math.ceil(budget * (split_index / full_unlabeled_size)) for split_index in split_indices]
                  
        beginning_split = 0
        
        selected_idx = []
        
        for i in range(self.num_partitions):
            
            end_split = split_indices[i]
            
            # Create a subset of the original unlabeled dataset as a partition.
            partition_index_list = list(range(beginning_split, end_split))
            current_partition = Subset(self.unlabeled_dataset, partition_index_list)
            
            # Calculate the budget for this partition
            if i == 0:
                partition_budget = partition_budget_splits[i]
            else:
                partition_budget = partition_budget_splits[i] - partition_budget_splits[i - 1]
                
            # With the new subset, create an instance of the wrapped strategy and call its select function.
            wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.model, self.target_classes, self.args)
            selected_partition_idxs = wrapped_strategy.select(partition_budget)
            
            # Use the partition_index_list to map the selected indices w/ respect to the current partition to the indices w/ respect to the dataset
            to_add_idxs = np.array(partition_index_list)[selected_partition_idxs]
            selected_idx.extend(to_add_idxs)
            beginning_split = end_split
            
        # Return the selected idx
        return selected_idx