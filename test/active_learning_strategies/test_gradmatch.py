from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.gradmatch_active import GradMatchActive
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestGradMatchActive(unittest.TestCase):
    
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
        self.args = {'batch_size': 1, 'device': device, 'loss': torch.nn.functional.cross_entropy}
        
    def test_fixed_weight_greedy_parallel(self):
        
        self.strategy = GradMatchActive(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args) 
        
        # A is now d x n, b is vector of length d
        matrix_A = torch.transpose(self.rand_unlabeled_dataset.wrapped_data_tensor, 0, 1).to(self.strategy.device)
        vector_b = torch.sum(self.rand_unlabeled_dataset.wrapped_data_tensor, dim=0).to(self.strategy.device)
        val_shape = self.rand_unlabeled_dataset.wrapped_data_tensor.shape[0]
        
        # Get the returned weight vector
        vector_x = self.strategy.fixed_weight_greedy_parallel(matrix_A, vector_b, val_shape, nnz=50)
        
        # Make sure vector is of correct length
        self.assertEqual(len(vector_x), self.num_unlabeled_points)
        
        # Make sure number of non-zero elements is equal to nnz (50)
        self.assertEqual(len(torch.nonzero(vector_x)), 50)
        
        # Make sure elements are either 0 or 1
        for x in vector_x:
            self.assertIn(x.item(), [0,1])
            
    def test_orthogonalmp_reg_parallel(self):
        
        self.strategy = GradMatchActive(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args) 
        
        # A is now d x n, b is vector of length d
        matrix_A = torch.transpose(self.rand_unlabeled_dataset.wrapped_data_tensor, 0, 1).to(self.strategy.device).detach()
        vector_b = torch.sum(self.rand_unlabeled_dataset.wrapped_data_tensor, dim=0).to(self.strategy.device).detach()
        
        # Do OMP with positive coefficients
        vector_x = self.strategy.orthogonalmp_reg_parallel(matrix_A, vector_b, nnz=50, positive=True)
        
        # Make sure vector is of correct length
        self.assertEqual(len(vector_x), self.num_unlabeled_points)
        
        # Make sure number of non-zero elements is less than or equal to nnz (50).
        # OMP minimizes the zero norm of vector_x, so there are not necessarily 
        # 50 returned non-zero components.
        self.assertLessEqual(len(torch.nonzero(vector_x)), 50)
        
        # Make sure elements are positive
        for x in vector_x:
            self.assertGreaterEqual(x.item(), 0)
        
        # Do OMP, but support negative coefficients and provide no nnz. Set lam to 5.
        vector_x = self.strategy.orthogonalmp_reg_parallel(matrix_A, vector_b, positive=False, lam=5)
        
        # Make sure vector is of correct length
        self.assertEqual(len(vector_x), self.num_unlabeled_points)
        
        # Make sure number of non-zero elements is less than or equal to the full size of the unlabeled dataset
        self.assertLessEqual(len(torch.nonzero(vector_x)), len(self.rand_unlabeled_dataset))
        
    def test_select(self):
        
        self.strategy = GradMatchActive(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args) 
        
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
        
    def test_select_weighted(self):
        
        self.strategy = GradMatchActive(self.rand_labeled_dataset, self.rand_unlabeled_dataset, self.mymodel, self.classes, self.args) 
        
        budget = 10
        idxs, weights = self.strategy.select(budget, True)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(self.strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))
        
        # Ensure weights are positive and number the same as the idxs
        self.assertEqual(len(weights), len(idxs))
        
        for weight in weights:
            self.assertGreater(weight, 0)