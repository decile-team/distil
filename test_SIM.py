CUDA_LAUNCH_BLOCKING="1"

import torch
import torchvision.datasets.cifar as cifar
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torchvision import transforms
from distil.utils.models.resnet import ResNet18
from PIL import Image
import sys
test = sys.argv[1]

from distil.scalable_active_learning_strategies.partition_strategy import PartitionStrategy
from distil.scalable_active_learning_strategies.badge import BADGE
from distil.scalable_active_learning_strategies.smi import SMI
from distil.scalable_active_learning_strategies.scg import SCG
from distil.scalable_active_learning_strategies.scmi import SCMI

class MyUnlabeledCIFAR10(Dataset):
    
    def __init__(self, cifar10_dataset):
        self.cifar10_dataset = cifar10_dataset
        
    def __getitem__(self, index):
        return self.cifar10_dataset[index][0]
    
    def __len__(self):
        return len(self.cifar10_dataset)

transform = transforms.Compose([transforms.ToTensor()])

root_dir = "/home/snk170001/bioml/dss/notebooks/data"
cifar10_dataset = cifar.CIFAR10(root_dir, transform = transform)
cifar10_test_dataset = cifar.CIFAR10(root_dir, transform = transform, train=False)

cifar10_labeled_dataset = Subset(cifar10_dataset, indices=[x for x in range(100)])
cifar10_query_dataset = Subset(MyUnlabeledCIFAR10(cifar10_dataset), indices=[x for x in range(100,200)])
cifar10_private_dataset = Subset(MyUnlabeledCIFAR10(cifar10_dataset), indices=[x for x in range(200,300)])
cifar10_unlabeled_dataset = Subset(MyUnlabeledCIFAR10(cifar10_dataset), indices=[x for x in range(300,4000)])

cifar10_model = ResNet18(10).to("cuda")
cifar10_model.load_state_dict(torch.load("/home/snk170001/bioml/dss/notebooks/weights/cifar10_ResNet18_0.01_50"))


args = {'batch_size': 20, 'device':'cuda', 'num_partitions':7, 'wrapped_strategy_class': None, 'smi_function':'fl2mi', 'scg_function':'flcg', "scmi_function":"flcmi"}
if(test=="scg"):
    args['wrapped_strategy_class'] = SCG
    my_cifar10_strategy = SCG(cifar10_labeled_dataset, cifar10_unlabeled_dataset, cifar10_query_dataset, cifar10_model, 10, args)
    my_part_cifar10_strategy = PartitionStrategy(cifar10_labeled_dataset, cifar10_unlabeled_dataset, cifar10_model, 10, args, None, cifar10_private_dataset)
elif(test=="smi"):
    args['wrapped_strategy_class'] = SMI
    print(args)
    my_cifar10_strategy = SMI(cifar10_labeled_dataset, cifar10_unlabeled_dataset, cifar10_private_dataset, cifar10_model, 10, args)
    my_part_cifar10_strategy = PartitionStrategy(cifar10_labeled_dataset, cifar10_unlabeled_dataset, cifar10_model, 10, args, cifar10_query_dataset, None)
elif(test=="scmi"):
    args['wrapped_strategy_class'] = SCMI
    my_cifar10_strategy = SCMI(cifar10_labeled_dataset, cifar10_unlabeled_dataset, cifar10_query_dataset, cifar10_private_dataset, cifar10_model, 10, args)
    my_part_cifar10_strategy = PartitionStrategy(cifar10_labeled_dataset, cifar10_unlabeled_dataset, cifar10_model, 10, args, cifar10_query_dataset, cifar10_private_dataset)


my_cifar10_strategy.update_data(cifar10_labeled_dataset, cifar10_unlabeled_dataset)
my_cifar10_strategy.update_model(cifar10_model)


indices = my_cifar10_strategy.select(100)
print(indices)
print(len(indices))

# args has 'wrapped_strategy_class' field.

indices = my_part_cifar10_strategy.select(100)
print(indices)
print(len(indices))