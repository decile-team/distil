from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.submod_sampling import SubmodularSampling
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
        self.num_labeled_points = 50
        rand_data_tensor = torch.randn((self.num_labeled_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_labeled_points,))
        self.rand_labeled_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # Create unlabeled dataset
        self.num_unlabeled_points = 1000
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
        
        # Try a different valid metric
        submod_args['metric'] = 'euclidean'
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        # Try an invalid metric
        submod_args['metric'] = 'invalid'
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
        
    def test_representation(self):
        
        budget = 10
        
        submod_args = {'submod': 'facility_location', 'representation': 'linear'}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        
        # Should pass the following representation choices
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        submod_args['representation'] = 'grad_bias'
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        submod_args['representation'] = 'grad_linear'
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        submod_args['representation'] = 'grad_bias_linear'
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        # Should fail an invalid metric
        submod_args['representation'] = 'invalid'
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
        
    def test_submod_facility_location(self):
        
        budget = 10
        
        submod_args = {'submod': 'facility_location'}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        
        # Should pass the following submod choices
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
    def test_submod_feature_based(self):
        
        budget = 10
        submod_args = {'submod': 'feature_based'}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
    
    def test_submod_graph_cut(self):
        
        budget = 10
        submod_args = {'submod': 'graph_cut', 'lambda_val': 1}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        del submod_args['lambda_val']
        
        # Should fail due to no lambda_val
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
    
    def test_submod_log_determinant(self):
        
        budget = 10
        submod_args = {'submod': 'log_determinant', 'lambda_val': 1}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
        del submod_args['lambda_val']
        
        # Should fail due to no lambda_val
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
    
    def test_submod_disparity_min(self):
        
        budget = 10
        submod_args = {'submod': 'disparity_min'}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
        
    def test_submod_disparity_sum(self):
        
        budget = 10
        submod_args = {'submod': 'disparity_sum'}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        strategy.select(budget)
    

    def test_submod_invalid(self):        
        
        budget = 10
        submod_args = {'submod': 'invalid'}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        self.assertRaises(type(BaseException()), strategy.select, budget)
        
    def test_select(self):
        
        budget = 10
        submod_args = {'submod': 'facility_location'}
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'submod_args': submod_args}
        strategy = SubmodularSampling(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        idxs = strategy.select(budget)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))