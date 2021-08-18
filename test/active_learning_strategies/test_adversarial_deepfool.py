from distil.utils.models.resnet import ResNet18
from distil.active_learning_strategies.adversarial_deepfool import AdversarialDeepFool
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch

class TestAdversarialDeepFool(unittest.TestCase):
    
    def setUp(self):
        
        # Create model
        self.classes = 10
        mymodel = ResNet18(self.classes)

        # Create labeled dataset            
        self.num_labeled_points = 100
        rand_data_tensor = torch.randn((self.num_labeled_points, 3, 32, 32), requires_grad=True)
        rand_label_tensor = torch.randint(low=0,high=self.classes,size=(self.num_labeled_points,))
        rand_labeled_dataset = MyLabeledDataset(rand_data_tensor, rand_label_tensor)
        
        # Create unlabeled dataset
        self.num_unlabeled_points = 500
        rand_data_tensor = torch.randn((self.num_unlabeled_points, 3, 32, 32), requires_grad=True)
        rand_unlabeled_dataset = MyUnlabeledDataset(rand_data_tensor)
        
        # Create args array
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        args = {'batch_size': 1, 'device': device, 'loss': torch.nn.functional.cross_entropy, 'max_iter': 30}
        
        self.strategy = AdversarialDeepFool(rand_labeled_dataset, rand_unlabeled_dataset, mymodel, self.classes, args)   
        
    def test_max_iter(self):
        
        self.assertEqual(30, self.strategy.max_iter)
        
    def test_deepfool(self):
        
        # Should not result in an error
        test_tensor = self.strategy.unlabeled_dataset[0].to(self.strategy.device)
        self.strategy.model = self.strategy.model.to(self.strategy.device)
        self.strategy.deepfool(test_tensor, self.strategy.model, self.strategy.target_classes)
        
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