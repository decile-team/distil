from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.smi import SMI
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestSMI(unittest.TestCase):
    
    def setUp(self):
        
        # Create model
        self.input_dimension = 50
        self.classes = 10
        self.hidden_units = 20
        self.mymodel = TwoLayerNet(self.input_dimension, self.classes, self.hidden_units)

        # Create labeled dataset            
        self.num_labeled_points = 50
        rand_data_tensor = torch.randn((self.num_labeled_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_labeled_points,))
        self.rand_labeled_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # Create unlabeled dataset
        self.num_unlabeled_points = 1000
        rand_data_tensor = torch.randn((self.num_unlabeled_points, self.input_dimension), requires_grad=True)
        self.rand_unlabeled_dataset = MyUnlabeledDataset(rand_data_tensor)
        
        # Create query dataset
        self.num_query_points = 100
        rand_data_tensor = torch.randn((self.num_query_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_query_points,))
        self.rand_query_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # FASS has many different initializations; create different strategy instantiations in tests
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_embedding_type(self):
        
        budget = 10
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'embedding_type': 'gradients', 'smi_function': 'fl2mi'}
        
        # Should work for the following choices
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        args['embedding_type'] = 'features'
        args['layer_name'] = 'linear1'
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        # Should fail; embedding type not supported
        args['embedding_type'] = 'invalid'
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
        
    def test_smi_function(self):
        
        budget = 10
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy}
        
        # Should work for the following choices
        args['smi_function'] = 'fl1mi'
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        args['smi_function'] = 'fl2mi'
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        args['smi_function'] = 'com'
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        args['smi_function'] = 'gcmi'
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        args['smi_function'] = 'logdetmi'
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        # Should fail; not supported
        args['smi_function'] = 'invalid'
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
        
    def test_select(self):
        
        budget = 10
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'smi_function': 'fl2mi'}
        strategy = SMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.mymodel, self.classes, args)
        idxs = strategy.select(budget)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))