from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.scmi import SCMI
from distil.active_learning_strategies.smi import SMI
from distil.active_learning_strategies.scg import SCG
from distil.active_learning_strategies.partition_strategy import PartitionStrategy
from distil.active_learning_strategies.badge import BADGE
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch
import time

class TestPartitionStrategy(unittest.TestCase):
    
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
        self.num_unlabeled_points = 15000
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
        
    def test_wrapped_strategy_class(self):
        
        # Should fail; no wrapped strategy
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy}
        self.assertRaises(type(BaseException()), PartitionStrategy, self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        
        # Should work; wrapped strategy
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': BADGE}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        
    def test_select_non_sim(self):
        
        budget = 30
        
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': BADGE, 'num_partitions':10}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        idxs = strategy.select(budget)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': BADGE, 'num_partitions':1}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args)
        idxs = strategy.select(budget)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
    def test_select_smi(self):
        
        budget = 30
        
        # Num partitions = 5 should be faster than no partition
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': SMI, 'num_partitions':10, 'smi_function':'fl1mi'}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args, self.rand_query_dataset)
        start_time = time.time()
        idxs = strategy.select(budget)
        end_time = time.time()
        part_time = end_time - start_time
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': SMI, 'num_partitions':1, 'smi_function':'fl1mi'}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args, self.rand_query_dataset)
        start_time = time.time()
        idxs = strategy.select(budget)
        end_time = time.time()
        non_part_time = end_time - start_time
        
        self.assertLess(part_time, non_part_time)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
    def test_select_scg(self):
        
        budget = 30
        
        # Num partitions = 5 should be faster than no partition
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': SCG, 'num_partitions':10, 'scg_function':'flcg'}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args, private_dataset=self.rand_query_dataset)
        start_time = time.time()
        idxs = strategy.select(budget)
        end_time = time.time()
        part_time = end_time - start_time
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': SCG, 'num_partitions':1, 'scg_function':'flcg'}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args, private_dataset=self.rand_query_dataset)
        start_time = time.time()
        idxs = strategy.select(budget)
        end_time = time.time()
        non_part_time = end_time - start_time
        
        self.assertLess(part_time, non_part_time)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
    def test_select_scmi(self):
        
        budget = 30
        
        # Num partitions = 5 should be faster than no partition
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': SCMI, 'num_partitions':10, 'scmi_function':'flcmi'}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args, self.rand_query_dataset, self.rand_private_dataset)
        start_time = time.time()
        idxs = strategy.select(budget)
        end_time = time.time()
        part_time = end_time - start_time
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
        args = {'batch_size': 1, 'device': self.device, 'loss': torch.nn.functional.cross_entropy, 'wrapped_strategy_class': SCMI, 'num_partitions':1, 'scmi_function':'flcmi'}
        strategy = PartitionStrategy(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, args, self.rand_query_dataset, self.rand_private_dataset)
        start_time = time.time()
        idxs = strategy.select(budget)
        end_time = time.time()
        non_part_time = end_time - start_time
        
        self.assertLess(part_time, non_part_time)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
    