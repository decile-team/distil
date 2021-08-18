from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.glister import GLISTER
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestGLISTER(unittest.TestCase):
    
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
        
        # Create val. dataset
        self.num_val_points = 5000
        rand_data_tensor = torch.randn((self.num_unlabeled_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_labeled_points,))
        self.rand_validation_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # Create args array
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.args = {'batch_size': 1, 'device': device, 'loss': torch.nn.functional.cross_entropy, 'lr': 0.01}
        
    def test_select_no_val(self):
        
        self.strategy = GLISTER(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args)  
        
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
        
    def test_select_val(self):
        
        self.strategy = GLISTER(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args, validation_dataset=self.rand_validation_dataset)  
        
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
        
    def test_select_reg_rand(self):
        
        self.strategy = GLISTER(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args, typeOf='Rand', lam=0.5)  

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
        
    def test_select_reg_div(self):
        
        self.strategy = GLISTER(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args, typeOf='Diversity', lam=1)  

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
        
    def test_select_reg_fac_loc(self):
        
        self.strategy = GLISTER(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args, typeOf='FacLoc', lam=1)  

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
        
    def test_select_kern_batch(self):
        
        self.strategy = GLISTER(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args, kernel_batch_size=100)  
        
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