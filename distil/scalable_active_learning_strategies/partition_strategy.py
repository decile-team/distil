import torch
import math
from distil.utils.data_handler import DataHandler_Points
from distil.scalable_active_learning_strategies.strategy import Strategy

class PartitionStrategy(Strategy):
    
    def __init__(self, labeled_dataloader, unlabeled_dataloader, net, nclasses, wrapped_strategy_class, args={}): #
        
        super(PartitionStrategy, self).__init__(labeled_dataloader, unlabeled_dataloader, net, nclasses, args)
        
        if "num_partitions" not in args:
            self.num_partitions = 1
        else:
            self.num_partitions = args["num_partitions"]
            
        self.wrapped_strategy_class = wrapped_strategy_class
        
    def retrieve_labeled_points_as_numpy(self):
        
        for batch_idx, (batch_data, batch_labels, element_idxs) in enumerate(self.labeled_dataloader):
            
            if batch_idx == 0:
                data_tensor = batch_data
                label_tensor = batch_labels
            else:
                data_tensor = torch.cat([data_tensor, batch_data], dim=0)
                label_tensor = torch.cat([label_tensor, batch_labels], dim=0)
        
        return data_tensor.cpu().numpy(), label_tensor.cpu().numpy()
        
    def select(self, budget):
        
        # The number of partitions should be less than or equal to the budget.
        # This is because the budget is evenly divided among the partitions (roughly),
        # so having a smaller budget than the number of partitions results in one or 
        # more partitions having a 0 budget, which should not happen.
        if self.num_partitions > budget:
            raise ValueError("Budget cannot be less than the number of partitions!")
        
        # Furthermore, the number of partitions cannot be more than the size of the unlabeled set
        if self.num_partitions > len(self.unlabeled_dataloader.dataset):
            raise ValueError("There cannot be more partitions than the size of the dataset!")
               
        # Calculate partition splits and budgets for each partition
        full_unlabeled_size = len(self.unlabeled_dataloader.dataset)
        split_indices = [math.ceil(full_unlabeled_size * ((1+x) / self.num_partitions)) for x in range(self.num_partitions)]        
        partition_budget_splits = [math.ceil(budget * (split_index / full_unlabeled_size)) for split_index in split_indices]
                    
        # Keep track of the total number of data instances evaluated and the index start of the current partition.
        evaluated_data_instances = 0
        partition_number = 0
        
        # Retrieve the labeled set as numpy arrays.
        labeled_data, labels = self.retrieve_labeled_points_as_numpy()
        
        # Declare unlabeled_tensor and index_map to placate code analysis. 
        # Also initialize that the unlabeled_tensor should be cleared.
        clear_unlabeled_tensor = True
        unlabeled_tensor = None
        index_map = None
        
        selected_idxs = []
        
        # Construct each partition by using the batch loader.
        for batch_idx, (unlabeled_batch, element_idxs) in enumerate(self.unlabeled_dataloader):
                
            # Check if this batch overlaps a partition cutoff
            if evaluated_data_instances + unlabeled_batch.shape[0] < split_indices[partition_number]:
                # Keep adding batches as normal.
                if clear_unlabeled_tensor:
                    unlabeled_tensor = unlabeled_batch
                    index_map = element_idxs
                    clear_unlabeled_tensor = False
                else:
                    unlabeled_tensor = torch.cat([unlabeled_tensor, unlabeled_batch], dim=0)            
                    index_map = torch.cat([index_map, element_idxs], dim=0)
        
                evaluated_data_instances += unlabeled_batch.shape[0]
            else:
                # This batch eventually crosses a partition boundary. Iteratively fragment this batch so that 
                # the rest of the partition is not overfilled.
                while evaluated_data_instances + unlabeled_batch.shape[0] >= split_indices[partition_number]:
                    
                    # Calculate remaining partition to take from the batch
                    remaining_point_count = split_indices[partition_number] - evaluated_data_instances
                
                    if remaining_point_count > 0:
                        # Fragment the batch and insert into the partition.
                        from_batch_to_add_unlabeled = unlabeled_batch[:remaining_point_count]
                        from_batch_to_add_idxs = element_idxs[:remaining_point_count]
                        unlabeled_batch = unlabeled_batch[remaining_point_count:]
                        element_idxs = element_idxs[remaining_point_count:]
                    
                        if clear_unlabeled_tensor:
                            unlabeled_tensor = from_batch_to_add_unlabeled
                            index_map = from_batch_to_add_idxs
                            clear_unlabeled_tensor = False
                        else:
                            unlabeled_tensor = torch.cat([unlabeled_tensor, from_batch_to_add_unlabeled], dim=0)            
                            index_map = torch.cat([index_map, from_batch_to_add_idxs], dim=0)
        
                        evaluated_data_instances += remaining_point_count
                   
                    # Now, the partition is complete. Create the wrapped strategy.
                    # Note: The numpy arrays already have the transform applied. DataHandler_Points does no 
                    # further transformation (identity), so it is used here (regardless of the dataset)
                    unlabeled_data = unlabeled_tensor.cpu().numpy()
                    wrapped_strategy = self.wrapped_strategy_class(labeled_data, labels, unlabeled_data, self.model, DataHandler_Points, self.target_classes, self.args)
                
                    # Call the wrapped strategy's select function with the partition budget, which comes from the partition number.
                    if partition_number == 0:
                        partition_budget = partition_budget_splits[partition_number]
                    else:
                        partition_budget = partition_budget_splits[partition_number] - partition_budget_splits[partition_number - 1]
                    
                    partition_idx = wrapped_strategy.select(partition_budget)
                
                    # The returned indices are with respect to the partition.
                    # They must be mapped back to the original indices.
                    converted_idxs = index_map[partition_idx]
                    selected_idxs.extend(converted_idxs)
                
                    # Lastly, clear unlabeled_tensor and increment partition number
                    partition_number += 1
                    clear_unlabeled_tensor = True

                    # Break out of the loop if the partition number is ge to number of partitions                    
                    if partition_number >= self.num_partitions:
                        break
                    
                # At this point, the last bit of the loaded batch can be added.
                if unlabeled_batch.shape[0] > 0:
                    if clear_unlabeled_tensor:
                        unlabeled_tensor = unlabeled_batch
                        index_map = element_idxs
                        clear_unlabeled_tensor = False
                    else:
                        unlabeled_tensor = torch.cat([unlabeled_tensor, unlabeled_batch], dim=0)            
                        index_map = torch.cat([index_map, element_idxs], dim=0)
        
                evaluated_data_instances += unlabeled_batch.shape[0]
        
        return selected_idxs