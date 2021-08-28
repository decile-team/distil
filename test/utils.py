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