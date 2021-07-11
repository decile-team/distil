from distil.utils.models.simple_net import TwoLayerNet
from distil.scalable_active_learning_strategies.submod_sampling import SubmodularSampling
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestSubmodularSampling(unittest.TestCase):
    
    def setUp(self):
        
        # Create model
        self.input_dimension = 50
        self.classes = 10
        self.hidden_units = 20
        self.mymodel = TwoLayerNet(self.input_dimension, self.classes, self.hidden_units)

        # Create labeled dataset            
        self.num_labeled_points = 1000
        rand_data_tensor = torch.randn((self.num_labeled_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_labeled_points,))
        self.rand_labeled_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # Create unlabeled dataset
        self.num_unlabeled_points = 10000
        rand_data_tensor = torch.randn((self.num_unlabeled_points, self.input_dimension), requires_grad=True)
        self.rand_unlabeled_dataset = MyUnlabeledDataset(rand_data_tensor)
        
        # FASS has many different initializations; create different strategy instantiations in tests
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        
    def test_metric(self):
        
        budget = 10
        
        submod_args = {'submod': 'facility_location', 'metric': 'cosine'}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        
        # Should pass; metric is okay
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)