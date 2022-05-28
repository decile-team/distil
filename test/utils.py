from torch.utils.data import Dataset

class MyLabeledDataset(Dataset):
            
    def __init__(self, wrapped_data_tensor, wrapped_label_tensor):
        self.wrapped_data_tensor = wrapped_data_tensor
        self.wrapped_label_tensor = wrapped_label_tensor
                
    def __getitem__(self, index):
        data = self.wrapped_data_tensor[index]
        label = self.wrapped_label_tensor[index]
        return data, label
            
    def __len__(self):
        return self.wrapped_label_tensor.shape[0]
            
class MyUnlabeledDataset(Dataset):
            
    def __init__(self, wrapped_data_tensor):
        self.wrapped_data_tensor = wrapped_data_tensor
                
    def __getitem__(self, index):
        data = self.wrapped_data_tensor[index]
        return data
            
    def __len__(self):
        return self.wrapped_data_tensor.shape[0]
    
class DictDatasetWrapper(Dataset):
    
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset
        
    def __getitem__(self, index):
        returned_instance = self.wrapped_dataset[index]
        
        if type(returned_instance) == tuple:
            return {"x": returned_instance[0], "labels": returned_instance[1]}
        else:
            return {"x": returned_instance}

    def __len__(self):
        return len(self.wrapped_dataset)