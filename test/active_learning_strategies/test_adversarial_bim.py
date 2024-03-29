from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.adversarial_bim import AdversarialBIM
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestAdversarialBIM(unittest.TestCase):
    
    def setUp(self):
        
        # Create model
        self.input_dimension = 50
        self.classes = 10
        self.hidden_units = 20
        mymodel = TwoLayerNet(self.input_dimension, self.classes, self.hidden_units)

        # Create labeled dataset            
        self.num_labeled_points = 1000
        rand_data_tensor = torch.randn((self.num_labeled_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_labeled_points,))
        rand_labeled_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # Create unlabeled dataset
        self.num_unlabeled_points = 10000
        rand_data_tensor = torch.randn((self.num_unlabeled_points, self.input_dimension), requires_grad=True)
        rand_unlabeled_dataset = MyUnlabeledDataset(rand_data_tensor)
        
        # Create args array
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        args = {'batch_size': 20, 'device': device, 'loss': torch.nn.functional.cross_entropy, 'eps': 0.04, 'verbose': True}
        
        self.strategy = AdversarialBIM(rand_labeled_dataset, rand_unlabeled_dataset, mymodel, self.classes, args)    
        
    def test_eps(self):
        
        self.assertEqual(0.04, self.strategy.eps)
        
    def test_cal_dis(self):
        
        # Should not result in an error
        test_tensor = self.strategy.unlabeled_dataset[0].to(self.strategy.device)
        self.strategy.model = self.strategy.model.to(self.strategy.device)
        self.strategy.cal_dis(test_tensor)
        
    def test_select(self):
        
        budget = 10
        idxs = self.strategy.select(budget)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(self.strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))