from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.margin_sampling import MarginSampling
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestMarginSampling(unittest.TestCase):
    
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
        args = {'batch_size': 1, 'device': device, 'loss': torch.nn.functional.cross_entropy}
        
        self.strategy = MarginSampling(rand_labeled_dataset, rand_unlabeled_dataset, mymodel, self.classes, args)  
        
    def test_acquire_scores(self):
        
        # Acquire scores for the entire dataset
        scores = self.strategy.acquire_scores(self.strategy.unlabeled_dataset)
        
        # Assert that there is a score for each point
        self.assertEqual(len(scores), len(self.strategy.unlabeled_dataset))
        
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