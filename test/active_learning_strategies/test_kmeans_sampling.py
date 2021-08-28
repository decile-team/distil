from distil.utils.models.simple_net import TwoLayerNet
from distil.active_learning_strategies.kmeans_sampling import KMeansSampling
from test.utils import MyLabeledDataset, MyUnlabeledDataset

import unittest
import torch
import math

class TestKMeansSampling(unittest.TestCase):
    
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
        args = {'batch_size': 1000, 'device': device, 'loss': torch.nn.functional.cross_entropy, 'representation': 'raw'}
        
        self.strategy = KMeansSampling(rand_labeled_dataset, rand_unlabeled_dataset, mymodel, self.classes, args)  

    def test_closest_distances(self):
        
        ground_set_tensor = torch.tensor([[1.,2.,3.], [2.,1.,2.], [3.,2.,1.], [10.,11.,8.], [9.,10.,11.], [11.,10.,9.]]).to(self.strategy.device)
        center_set_tensor = torch.tensor([[2.,2.,2.],[10.,10.,10.]]).to(self.strategy.device)
        
        ground_set = MyUnlabeledDataset(ground_set_tensor)
        
        ground_set_distances_to_closest_center, ground_set_closest_center_indices = self.strategy.get_closest_distances(ground_set, center_set_tensor)
        
        # Check that the length of each tensor matches the length of the ground set
        self.assertEqual(len(ground_set_distances_to_closest_center), len(ground_set))
        self.assertEqual(len(ground_set_closest_center_indices), len(ground_set))
        
        # Check that the first three points in the ground set are closest to the first center and that the last
        # three are closest to the second center
        self.assertEqual(ground_set_closest_center_indices[0], 0)
        self.assertEqual(ground_set_closest_center_indices[1], 0)
        self.assertEqual(ground_set_closest_center_indices[2], 0)
        self.assertEqual(ground_set_closest_center_indices[3], 1)
        self.assertEqual(ground_set_closest_center_indices[4], 1)
        self.assertEqual(ground_set_closest_center_indices[5], 1)
        
        # Check that the distances are correct
        self.assertAlmostEqual(ground_set_distances_to_closest_center[0].item(), math.sqrt(2))
        self.assertAlmostEqual(ground_set_distances_to_closest_center[1].item(), 1.)
        self.assertAlmostEqual(ground_set_distances_to_closest_center[2].item(), math.sqrt(2))
        self.assertAlmostEqual(ground_set_distances_to_closest_center[3].item(), math.sqrt(5))
        self.assertAlmostEqual(ground_set_distances_to_closest_center[4].item(), math.sqrt(2))
        self.assertAlmostEqual(ground_set_distances_to_closest_center[5].item(), math.sqrt(2))
        
    def test_closest_distances2(self):
        
        center_set_tensor = torch.tensor([[0.25 for x in range(self.input_dimension)], [0.5 for x in range(self.input_dimension)], [0.75 for x in range(self.input_dimension)]]).to(self.strategy.device)
        
        ground_set = self.strategy.unlabeled_dataset
        
        ground_set_distances_to_closest_center, ground_set_closest_center_indices = self.strategy.get_closest_distances(ground_set, center_set_tensor)
        
        # Check that the length of each tensor matches the length of the ground set
        self.assertEqual(len(ground_set_distances_to_closest_center), len(ground_set))
        self.assertEqual(len(ground_set_closest_center_indices), len(ground_set))
        
        # Check that all distances are greater than 0
        for distance in ground_set_distances_to_closest_center:
            self.assertGreaterEqual(distance, 0)
            
        # Check that all center indices are in range
        for index in ground_set_closest_center_indices:
            self.assertIn(index, list(range(len(center_set_tensor))))
            
    def test_closest_distances_linear(self):
        
        # Use last linear layer embedding
        self.strategy.representation = 'linear'
        
        # Clear first linear layer bias, set first linear layer to 1s
        # Ensures that, if x is embedding, x_i = sum(input_i)
        change_model = self.strategy.model
        change_model.linear1.weight.data = torch.ones_like(change_model.linear1.weight.data)
        change_model.linear1.bias.data = torch.zeros_like(change_model.linear1.bias.data)
        self.strategy.update_model(change_model)
        
        # Test above assumption
        test_tensor = torch.tensor([[1. for x in range(self.input_dimension)]])
        prediction, last = self.strategy.model(test_tensor, last=True)
        self.assertEqual(len(last[0]), self.hidden_units)
        for coordinate in last[0]:
            self.assertEqual(coordinate.item(), self.input_dimension)
            
        # Now, do the work above except with the linear embedding argument
        ground_set_tensor = torch.tensor([[1. for x in range(self.input_dimension)], [2. for x in range(self.input_dimension)], [3. for x in range(self.input_dimension)]]).to(self.strategy.device)
        center_set_tensor = torch.tensor([[float(self.input_dimension) for x in range(self.hidden_units)],
                                          [2. * self.input_dimension for x in range(self.hidden_units)]]).to(self.strategy.device)
        
        ground_set = MyUnlabeledDataset(ground_set_tensor)
        ground_set_distances_to_closest_center, ground_set_closest_center_indices = self.strategy.get_closest_distances(ground_set, center_set_tensor)
        
        # Check that the length of each tensor matches the length of the ground set
        self.assertEqual(len(ground_set_distances_to_closest_center), len(ground_set))
        self.assertEqual(len(ground_set_closest_center_indices), len(ground_set))
        
        # Check that the first three points in the ground set are closest to the first center and that the last
        # three are closest to the second center
        self.assertEqual(ground_set_closest_center_indices[0], 0)
        self.assertEqual(ground_set_closest_center_indices[1], 1)
        self.assertEqual(ground_set_closest_center_indices[2], 1)
        
        # Check that the distances are correct
        self.assertAlmostEqual(ground_set_distances_to_closest_center[0].item(), 0)
        self.assertAlmostEqual(ground_set_distances_to_closest_center[1].item(), 0)
        self.assertAlmostEqual(ground_set_distances_to_closest_center[2].item(), math.sqrt(self.input_dimension * self.input_dimension * self.hidden_units), places=5)
        
    def test_closest_distances_linear2(self):
        
        # Use last linear layer embedding
        self.strategy.representation = 'linear'
        
        # Clear first linear layer bias, set first linear layer to 1s
        # Ensures that, if x is embedding, x_i = sum(input_i)
        change_model = self.strategy.model
        change_model.linear1.weight.data = torch.ones_like(change_model.linear1.weight.data)
        change_model.linear1.bias.data = torch.zeros_like(change_model.linear1.bias.data)
        self.strategy.update_model(change_model)
        
        # Test above assumption
        test_tensor = torch.tensor([[1. for x in range(self.input_dimension)]])
        prediction, last = self.strategy.model(test_tensor, last=True)
        self.assertEqual(len(last[0]), self.hidden_units)
        for coordinate in last[0]:
            self.assertEqual(coordinate.item(), self.input_dimension)
            
        # Now, do the work above except with the linear embedding argument
        center_set_tensor = torch.tensor([[0.25 for x in range(self.hidden_units)], [0.5 for x in range(self.hidden_units)], [0.75 for x in range(self.hidden_units)]]).to(self.strategy.device)
        
        ground_set = self.strategy.unlabeled_dataset
        
        ground_set_distances_to_closest_center, ground_set_closest_center_indices = self.strategy.get_closest_distances(ground_set, center_set_tensor)
        
        # Check that the length of each tensor matches the length of the ground set
        self.assertEqual(len(ground_set_distances_to_closest_center), len(ground_set))
        self.assertEqual(len(ground_set_closest_center_indices), len(ground_set))
        
        # Check that all distances are greater than 0
        for distance in ground_set_distances_to_closest_center:
            self.assertGreaterEqual(distance, 0)
            
        # Check that all center indices are in range
        for index in ground_set_closest_center_indices:
            self.assertIn(index, list(range(len(center_set_tensor))))
            
    def test_kmeans_plusplus(self):
        
        num_centers = 300
        centers_idx = self.strategy.kmeans_plusplus(num_centers)
        
        # Ensure there are 10 centers whose indices are within the range of the unlabeled dataset
        self.assertEqual(len(centers_idx), num_centers)
        for idx in centers_idx:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.num_unlabeled_points)
        
        # Ensure unique centers are chosen
        self.assertEqual(len(set(centers_idx)), len(centers_idx))
        
        # Run kmeans plus plus again; should get a different initialization after at least a couple tries
        new_centers_idx = self.strategy.kmeans_plusplus(num_centers)
        self.assertEqual(len(new_centers_idx), num_centers)
        self.assertEqual(len(set(centers_idx)), len(centers_idx))
        for idx in new_centers_idx:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.num_unlabeled_points)
        
        different = False
        for i in range(5):
            for x,y in zip(centers_idx,new_centers_idx):
                if x != y:
                    different = True
            if not different:
                new_centers_idx = self.strategy.kmeans_plusplus(num_centers)
                self.assertEqual(len(new_centers_idx), num_centers)
                self.assertEqual(len(set(centers_idx)), len(centers_idx))
                for idx in new_centers_idx:
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, self.num_unlabeled_points)
        self.assertTrue(different)
                    
    def test_kmeans_plusplus_linear(self):
     
        # Repeat kmeans++ with the linear representation instead.
        self.strategy.representation = 'linear'
        
        num_centers = 300
        centers_idx = self.strategy.kmeans_plusplus(num_centers)
        
        # Ensure there are 10 centers whose indices are within the range of the unlabeled dataset
        self.assertEqual(len(centers_idx), num_centers)
        for idx in centers_idx:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.num_unlabeled_points)
        
        # Ensure unique centers are chosen
        self.assertEqual(len(set(centers_idx)), len(centers_idx))
        
        # Run kmeans plus plus again; should get a different initialization after at least a couple tries
        new_centers_idx = self.strategy.kmeans_plusplus(num_centers)
        self.assertEqual(len(new_centers_idx), num_centers)
        self.assertEqual(len(set(centers_idx)), len(centers_idx))
        for idx in new_centers_idx:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.num_unlabeled_points)
        
        different = False
        for i in range(5):
            for x,y in zip(centers_idx,new_centers_idx):
                if x != y:
                    different = True
            if not different:
                new_centers_idx = self.strategy.kmeans_plusplus(num_centers)
                self.assertEqual(len(new_centers_idx), num_centers)
                self.assertEqual(len(set(centers_idx)), len(centers_idx))
                for idx in new_centers_idx:
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, self.num_unlabeled_points)
        self.assertTrue(different)
        
    def test_kmeans_calculate_means(self):
        
        new_dataset_tensor = torch.tensor([[float(y) for x in range(self.input_dimension)] for y in range(10)]).to(self.strategy.device)
        new_dataset = MyUnlabeledDataset(new_dataset_tensor)
        
        self.strategy.unlabeled_dataset = new_dataset
        
        clusters = [[0,1,2,3,4],[5,6,7],[8,9]]
        
        means = self.strategy.kmeans_calculate_means(clusters)
        
        # Ensure that there are 3 means
        self.assertEqual(len(means), len(clusters))
        
        # Ensure clusters match embedding
        for mean in means:
            self.assertEqual(len(mean), self.input_dimension)
            
        # Ensure first mean has elements equal to 2.
        mean = means[0]
        for element in mean:
            self.assertAlmostEqual(element.item(), 2.)
            
        # Ensure second mean has elements equal to 6.
        mean = means[1]
        for element in mean:
            self.assertAlmostEqual(element.item(), 6.)
            
        # Ensure third mean has elements equal to 8.5.
        mean = means[2]
        for element in mean:
            self.assertAlmostEqual(element.item(), 8.5)
            
    def test_kmeans_calculate_means_linear(self):
        
        # Use last linear layer embedding
        self.strategy.representation = 'linear'
        
        # Clear first linear layer bias, set first linear layer to 1s
        # Ensures that, if x is embedding, x_i = sum(input_i)
        change_model = self.strategy.model
        change_model.linear1.weight.data = torch.ones_like(change_model.linear1.weight.data)
        change_model.linear1.bias.data = torch.zeros_like(change_model.linear1.bias.data)
        self.strategy.update_model(change_model)
        
        # Test above assumption
        test_tensor = torch.tensor([[1. for x in range(self.input_dimension)]])
        prediction, last = self.strategy.model(test_tensor, last=True)
        self.assertEqual(len(last[0]), self.hidden_units)
        for coordinate in last[0]:
            self.assertEqual(coordinate.item(), self.input_dimension)
            
        # Now, do test on linear embedding space
        new_dataset_tensor = torch.tensor([[float(y) for x in range(self.input_dimension)] for y in range(10)]).to(self.strategy.device)
        new_dataset = MyUnlabeledDataset(new_dataset_tensor)
        
        self.strategy.unlabeled_dataset = new_dataset
        
        clusters = [[0,1,2,3,4],[5,6,7],[8,9]]
        
        means = self.strategy.kmeans_calculate_means(clusters)
        
        # Ensure that there are 3 means
        self.assertEqual(len(means), len(clusters))
        
        # Ensure clusters match embedding
        for mean in means:
            self.assertEqual(len(mean), self.hidden_units)
            
        # Ensure first mean has elements equal to 2.
        mean = means[0]
        for element in mean:
            self.assertAlmostEqual(element.item(), 2. * self.input_dimension)
            
        # Ensure second mean has elements equal to 6.
        mean = means[1]
        for element in mean:
            self.assertAlmostEqual(element.item(), 6. * self.input_dimension)
            
        # Ensure third mean has elements equal to 8.5.
        mean = means[2]
        for element in mean:
            self.assertAlmostEqual(element.item(), 8.5 * self.input_dimension)
            
    def test_kmeans_calculate_clusters(self):
        
        # For testing purposes, make the unlabeled set predictable for testing purposes
        new_dataset_tensor = torch.tensor([[float(y) for x in range(self.input_dimension)] for y in range(10)]).to(self.strategy.device)
        new_dataset = MyUnlabeledDataset(new_dataset_tensor)
        self.strategy.unlabeled_dataset = new_dataset
        
        # Create two centers
        new_center_tensor = torch.tensor([[1. for x in range(self.input_dimension)], [8. for x in range(self.input_dimension)]]).to(self.strategy.device)
        
        # Get new clusters
        clusters = self.strategy.kmeans_calculate_clusters(new_center_tensor)
        
        # Ensure there are two clusters
        self.assertEqual(len(clusters), len(new_center_tensor))
        
        # Ensure the first 5 elements of the unlabeled dataset are in the first, the rest in the second
        self.assertIn(0, clusters[0])
        self.assertIn(1, clusters[0])
        self.assertIn(2, clusters[0])
        self.assertIn(3, clusters[0])
        self.assertIn(4, clusters[0])
        self.assertIn(5, clusters[1])
        self.assertIn(6, clusters[1])
        self.assertIn(7, clusters[1])
        self.assertIn(8, clusters[1])
        self.assertIn(9, clusters[1])
        
        # Ensure no other elements are outside of the dataset range
        for cluster in clusters:
            for index in cluster:
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(self.strategy.unlabeled_dataset))
        
        # Ensure all indices belong to a cluster
        index_list = []
        for cluster in clusters:
            index_list.extend(cluster)            
        self.assertEqual(set(index_list), set(range(len(self.strategy.unlabeled_dataset))))
        
        # Ensure clusters do not overlap
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i == j:
                    continue
                self.assertEqual(len(set(cluster1).intersection(set(cluster2))), 0)
                
    def test_kmeans_clustering(self):
        
        num_centers = 10
        best_centers = self.strategy.kmeans_clustering(num_centers)
        
        # Ensure that there are 10 centers and correct embedding
        self.assertEqual(best_centers.shape[0], num_centers)
        self.assertEqual(best_centers.shape[1], self.input_dimension)
        
    def test_kmeans_clustering_linear(self):
        
        self.strategy.representation = 'linear'
        
        num_centers = 10
        best_centers = self.strategy.kmeans_clustering(num_centers)
        
        # Ensure that there are 10 centers and correct embedding
        self.assertEqual(best_centers.shape[0], num_centers)
        self.assertEqual(best_centers.shape[1], self.hidden_units)
        
    def test_select(self):
        
        budget = 100
        idxs = self.strategy.select(budget)
        
        # Ensure that indices are within the range spanned by the unlabeled dataset
        for idx in idxs:
            self.assertLess(idx, len(self.strategy.unlabeled_dataset))
            self.assertGreaterEqual(idx, 0)
            
        # Ensure that `budget` idx were returned
        self.assertEqual(budget, len(idxs))
        
        # Ensure that no point is selected multiple times
        self.assertEqual(len(idxs), len(set(idxs)))