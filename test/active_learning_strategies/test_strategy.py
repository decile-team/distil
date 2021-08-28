from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.strategy import Strategy
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestStrategy(unittest.TestCase):
    
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
        
        self.strategy = Strategy(rand_labeled_dataset, rand_unlabeled_dataset, mymodel, self.classes, args)    
    
    def test_update_data(self):
        
        old_unlabeled_dataset = self.strategy.unlabeled_dataset
        old_labeled_dataset = self.strategy.labeled_dataset
        
        # Create new labeled dataset
        rand_l_data_tensor = torch.randn((self.num_labeled_points, self.input_dimension), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_labeled_points,))
        rand_labeled_dataset = MyLabeledDataset(rand_l_data_tensor, rand_label_tensor)
        
        # Create unlabeled dataset
        rand_data_tensor = torch.randn((self.num_unlabeled_points, self.input_dimension), requires_grad=True)
        rand_unlabeled_dataset = MyUnlabeledDataset(rand_data_tensor)
        
        # Update the data
        self.strategy.update_data(rand_labeled_dataset, rand_unlabeled_dataset)
        
        # Make sure the tensors are different
        self.assertFalse(torch.equal(self.strategy.labeled_dataset.wrapped_data_tensor, old_labeled_dataset.wrapped_data_tensor))
        self.assertFalse(torch.equal(self.strategy.labeled_dataset.wrapped_label_tensor, old_labeled_dataset.wrapped_label_tensor))
        self.assertFalse(torch.equal(self.strategy.unlabeled_dataset.wrapped_data_tensor, old_unlabeled_dataset.wrapped_data_tensor))
        
        # Make sure the updated datasets are the same
        self.assertTrue(torch.equal(self.strategy.labeled_dataset.wrapped_data_tensor, rand_l_data_tensor))
        self.assertTrue(torch.equal(self.strategy.labeled_dataset.wrapped_label_tensor, rand_label_tensor))
        self.assertTrue(torch.equal(self.strategy.unlabeled_dataset.wrapped_data_tensor, rand_data_tensor))
        
        # Update works; revert back to old datasets
        self.strategy.update_data(old_labeled_dataset, old_unlabeled_dataset)
        
        # Make sure the tensors are the same
        self.assertTrue(torch.equal(self.strategy.labeled_dataset.wrapped_data_tensor, old_labeled_dataset.wrapped_data_tensor))
        self.assertTrue(torch.equal(self.strategy.labeled_dataset.wrapped_label_tensor, old_labeled_dataset.wrapped_label_tensor))
        self.assertTrue(torch.equal(self.strategy.unlabeled_dataset.wrapped_data_tensor, old_unlabeled_dataset.wrapped_data_tensor))
        
    def test_update_model(self):
        
        # Create a new model with two extra hidden units
        old_model = self.strategy.model
        my_model = TwoLayerNet(self.input_dimension, self.classes, self.hidden_units + 2)
        
        self.strategy.update_model(my_model)
        
        # Make sure the models are not equal
        self.assertNotEqual(old_model, self.strategy.model)
        
        # Update works; revert back to old model
        self.strategy.update_model(old_model)
        
        # Make sure the models are equal
        self.assertEqual(self.strategy.model, old_model)
        
    def test_predict(self):
        
        # Predict labels for the unlabeled dataset
        predicted_labels = self.strategy.predict(self.strategy.unlabeled_dataset)
        
        # Ensure the same number of labels exist as the number of points
        self.assertEqual(len(predicted_labels), len(self.strategy.unlabeled_dataset))
        
        # Ensure none of the predicted labels are outside the expected range
        for predicted_label in predicted_labels:
            self.assertLess(predicted_label, self.strategy.target_classes)
            self.assertGreaterEqual(predicted_label, 0)
            
    def test_predict_prob(self):
        
        # Predict probabilities for the unlabeled dataset
        predict_probs = self.strategy.predict_prob(self.strategy.unlabeled_dataset)
        
        # Ensure the same number of probability vectors and number of probabilities
        self.assertEqual(predict_probs.shape[0], len(self.strategy.unlabeled_dataset))
        self.assertEqual(predict_probs.shape[1], self.strategy.target_classes)
        
        # Ensure probabilities sum to 1
        for predicted_prob_vector in predict_probs:
            self.assertAlmostEqual(predicted_prob_vector.sum().item(), 1, places=6)
            
        # Ensure probabilities are geq 0, leq 1
        for predicted_prob_vector in predict_probs:
            for predicted_prob in predicted_prob_vector:
                self.assertLessEqual(predicted_prob, 1)
                self.assertGreaterEqual(predicted_prob, 0)
    
    def test_predict_prob_dropout(self):
        
        # Predict probabilities for the unlabeled dataset
        predict_probs = self.strategy.predict_prob_dropout(self.strategy.unlabeled_dataset, n_drop=5)
        
        # Ensure the same number of probability vectors and number of probabilities
        self.assertEqual(predict_probs.shape[0], len(self.strategy.unlabeled_dataset))
        self.assertEqual(predict_probs.shape[1], self.strategy.target_classes)
        
        # Ensure probabilities sum to 1
        for predicted_prob_vector in predict_probs:
            self.assertAlmostEqual(predicted_prob_vector.sum().item(), 1, places=6)
            
        # Ensure probabilities are geq 0, leq 1
        for predicted_prob_vector in predict_probs:
            for predicted_prob in predicted_prob_vector:
                self.assertLessEqual(predicted_prob, 1)
                self.assertGreaterEqual(predicted_prob, 0)
                
    def test_predict_prob_dropout_split(self):
        
        # Predict probabilities for the unlabeled dataset
        n_drop = 5
        predict_probs = self.strategy.predict_prob_dropout_split(self.strategy.unlabeled_dataset, n_drop=n_drop)
        
        # Ensure the same number of probability vectors and number of probabilities and number of dropout samples
        self.assertEqual(predict_probs.shape[0], n_drop)
        self.assertEqual(predict_probs.shape[1], len(self.strategy.unlabeled_dataset))
        self.assertEqual(predict_probs.shape[2], self.strategy.target_classes)
        
        # Ensure probabilities sum to 1
        for predict_prob_dropout in predict_probs:
            for predicted_prob_vector in predict_prob_dropout:
                self.assertAlmostEqual(predicted_prob_vector.sum().item(), 1, places=6)
            
        # Ensure probabilities are geq 0, leq 1
        for predict_prob_dropout in predict_probs:
            for predicted_prob_vector in predict_prob_dropout:
                for predicted_prob in predicted_prob_vector:
                    self.assertLessEqual(predicted_prob, 1)
                    self.assertGreaterEqual(predicted_prob, 0)
                    
    def test_get_embedding(self):
        
        # Get a last linear layer embedding
        embedding = self.strategy.get_embedding(self.strategy.unlabeled_dataset)
    
        # Ensure embedding has number of points equal to the unlabeled dataset
        self.assertEqual(embedding.shape[0], len(self.strategy.unlabeled_dataset))

        # Ensure embedding has number of features equal to the embedding of the model
        self.assertEqual(embedding.shape[1], self.strategy.model.get_embedding_dim())
        
    def test_get_grad_embedding(self):
        
        # Get grad embedding (bias)
        bias_grad_embedding = self.strategy.get_grad_embedding(self.strategy.unlabeled_dataset, predict_labels=True, grad_embedding_type='bias')
        
        # Ensure grad embedding has correct number of points / dimension
        self.assertEqual(bias_grad_embedding.shape[0], len(self.strategy.unlabeled_dataset))
        self.assertEqual(bias_grad_embedding.shape[1], self.strategy.target_classes)
        
        # Get grad embedding (linear)
        linear_grad_embedding = self.strategy.get_grad_embedding(self.strategy.unlabeled_dataset, predict_labels=True, grad_embedding_type='linear')
        
        # Ensure grad embedding has correct number of points / dimension
        self.assertEqual(linear_grad_embedding.shape[0], len(self.strategy.unlabeled_dataset))
        self.assertEqual(linear_grad_embedding.shape[1], self.strategy.model.get_embedding_dim() * self.strategy.target_classes)
        
        # Get grad embedding (bias_linear)
        bias_linear_grad_embedding = self.strategy.get_grad_embedding(self.strategy.unlabeled_dataset, predict_labels=True, grad_embedding_type='bias_linear')
        
        # Ensure grad embedding has correct number of points / dimension
        self.assertEqual(bias_linear_grad_embedding.shape[0], len(self.strategy.unlabeled_dataset))
        self.assertEqual(bias_linear_grad_embedding.shape[1], self.strategy.model.get_embedding_dim() * self.strategy.target_classes + self.strategy.target_classes)

        # Get grad embedding on labeled dataset (bias)
        bias_grad_embedding = self.strategy.get_grad_embedding(self.strategy.labeled_dataset, predict_labels=False, grad_embedding_type='bias')
        
        # Ensure grad embedding has correct number of points / dimension
        self.assertEqual(bias_grad_embedding.shape[0], len(self.strategy.labeled_dataset))
        self.assertEqual(bias_grad_embedding.shape[1], self.strategy.target_classes)
        
        # Get grad embedding on labeled dataset (linear)
        linear_grad_embedding = self.strategy.get_grad_embedding(self.strategy.labeled_dataset, predict_labels=False, grad_embedding_type='linear')
        
        # Ensure grad embedding has correct number of points / dimension
        self.assertEqual(linear_grad_embedding.shape[0], len(self.strategy.labeled_dataset))
        self.assertEqual(linear_grad_embedding.shape[1], self.strategy.model.get_embedding_dim() * self.strategy.target_classes)
        
        # Get grad embedding on labeled dataset (bias_linear)
        bias_linear_grad_embedding = self.strategy.get_grad_embedding(self.strategy.labeled_dataset, predict_labels=False, grad_embedding_type='bias_linear')
        
        # Ensure grad embedding has correct number of points / dimension
        self.assertEqual(bias_linear_grad_embedding.shape[0], len(self.strategy.labeled_dataset))
        self.assertEqual(bias_linear_grad_embedding.shape[1], self.strategy.model.get_embedding_dim() * self.strategy.target_classes + self.strategy.target_classes)       
        
        # Make sure that ValueError is raised on invalid grad_embedding_type
        with self.assertRaises(ValueError):
            self.strategy.get_grad_embedding(self.strategy.unlabeled_dataset, predict_labels=True, grad_embedding_type='invalid_type')
            
if __name__ == "__main__":
    unittest.main()