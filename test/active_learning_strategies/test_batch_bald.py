from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.batch_bald import BatchBALDDropout
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestBatchBALDDropout(unittest.TestCase):
    
    def setUp(self):
        
        # Create model
        self.input_dimension = 50
        self.classes = 10
        self.hidden_units = 20
        mymodel = TwoLayerNet(self.input_dimension, self.classes, self.hidden_units)

        # Create labeled dataset            
        self.num_labeled_points = 100
        rand_data_tensor = torch.randn((self.num_labeled_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_labeled_points,))
        rand_labeled_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # Create unlabeled dataset
        self.num_unlabeled_points = 500
        rand_data_tensor = torch.randn((self.num_unlabeled_points, self.input_dimension), requires_grad=True)
        rand_unlabeled_dataset = MyUnlabeledDataset(rand_data_tensor)
        
        # Create args array
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        args = {'batch_size': 5, 'device': device, 'loss': torch.nn.functional.cross_entropy, 'eps': 0.04, 'mod_inject': 'linear2'}
        
        self.strategy = BatchBALDDropout(rand_labeled_dataset, rand_unlabeled_dataset, mymodel, self.classes, args)
        
    def test_do_MC_dropout_before_linear(self):
        
        n_drop = 5
        self.strategy.model = self.strategy.model.to(self.strategy.device)
        
        # Attempt to inject dropout at the specified layer of the TwoLayerNet
        mc_dropout_injection_samples = self.strategy.do_MC_dropout_before_linear(self.strategy.unlabeled_dataset, n_drop)
        
         # Ensure the same number of probability vectors and number of probabilities and number of dropout samples
        self.assertEqual(mc_dropout_injection_samples.shape[1], n_drop)
        self.assertEqual(mc_dropout_injection_samples.shape[0], len(self.strategy.unlabeled_dataset))
        self.assertEqual(mc_dropout_injection_samples.shape[2], self.strategy.target_classes)
        
        # Ensure probabilities sum to 1
        for predict_prob_dropout in mc_dropout_injection_samples:
            for predicted_prob_vector in predict_prob_dropout:
                self.assertAlmostEqual(predicted_prob_vector.sum().item(), 1, places=6)
            
        # Ensure probabilities are geq 0, leq 1
        for predict_prob_dropout in mc_dropout_injection_samples:
            for predicted_prob_vector in predict_prob_dropout:
                for predicted_prob in predicted_prob_vector:
                    self.assertLessEqual(predicted_prob, 1)
                    self.assertGreaterEqual(predicted_prob, 0)
                    
        # Ensure probabilities are not all the same
        for predict_prob_dropout in mc_dropout_injection_samples:
            same_samples = True
            first_mc_sample = predict_prob_dropout[0]
            for other_mc_sample in predict_prob_dropout:
                is_close_vector = torch.isclose(first_mc_sample, other_mc_sample)
                for is_close_component in is_close_vector:
                    if not is_close_component:
                        same_samples = False
                
            # Ensure there were different samples
            self.assertFalse(same_samples)
    
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
    