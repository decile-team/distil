from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.scmi import SCMI
from test.utils import MyLabeledDataset, MyUnlabeledDataset, DictDatasetWrapper

import unittest
import torch

class TestSCMI(unittest.TestCase):
    
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
        
        # Create private dataset
        self.num_private_points = 100
        rand_data_tensor = torch.randn((self.num_private_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_private_points,))
        self.rand_private_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # FASS has many different initializations; create different strategy instantiations in tests
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_embedding_type(self):
        
        budget = 10
        args = {'batch_size': 20, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'embedding_type': 'gradients', 'scmi_function': 'flcmi'}
        
        # Should work for the following choices
        strategy = SCMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.rand_private_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        args['embedding_type'] = 'features'
        args['layer_name'] = 'linear1'
        strategy = SCMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.rand_private_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        # Should fail; embedding type not supported
        args['embedding_type'] = 'invalid'
        strategy = SCMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.rand_private_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
        
    def test_smi_function(self):
        
        budget = 10
        args = {'batch_size': 20, 'device': self.device, 'loss': torch.nn.functional.cross_entropy}
        
        # Should work for the following choices
        args['scmi_function'] = 'flcmi'
        strategy = SCMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.rand_private_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)

        args['scmi_function'] = 'logdetcmi'
        strategy = SCMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.rand_private_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        # Should fail; not supported
        args['scmi_function'] = 'invalid'
        strategy = SCMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.rand_private_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
        
    def test_select(self):
        
        budget = 10
        args = {'batch_size': 20, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'scmi_function': 'flcmi'}
        strategy = SCMI(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.rand_query_dataset, self.rand_private_dataset, self.mymodel, self.classes, args)
        idxs = strategy.select(budget)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
    def test_select_dict(self):
        
        budget = 10
        args = {'batch_size': 20, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'scmi_function': 'flcmi'}
        strategy = SCMI(DictDatasetWrapper(self.rand_labeled_dataset), DictDatasetWrapper(self.rand_unlabeled_dataset), DictDatasetWrapper(self.rand_query_dataset), DictDatasetWrapper(self.rand_private_dataset), self.mymodel, self.classes, args)
        idxs = strategy.select(budget)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))