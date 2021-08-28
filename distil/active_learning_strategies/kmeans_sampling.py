from .strategy import Strategy
from torch.utils.data import Subset, DataLoader, Dataset
from numpy import random
import torch

# Used to calculate embeddings
class CustomTensorDataset(Dataset):
            
    def __init__(self, wrapped_tensor):
        self.wrapped_tensor = wrapped_tensor
                
    def __getitem__(self, index):
        return self.wrapped_tensor[index]
            
    def __len__(self):
        return self.wrapped_tensor.shape[0]

class KMeansSampling(Strategy):
    
    """
    Implements KMeans Sampling selection strategy, the last layer embeddings are calculated for all the unlabeled data points. 
    Then the KMeans clustering algorithm is run over these embeddings with the number of clusters equal to the budget. 
    Then the distance is calculated for all the points from their respective centers. From each cluster, the point closest to 
    the center is selected to be labeled for the next iteration. Since the number of centers are equal to the budget, selecting 
    one point from each cluster satisfies the total number of data points to be selected in one iteration.
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled training dataset
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    args: dict
        Specify additional parameters
        
        - **batch_size**: Batch size to be used inside strategy class (int, optional)
        - **device**: The device that this strategy class should use for computation (string, optional)
        - **loss**: The loss that should be used for relevant computations (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
        - **rand_seed**: Specifies a seed for the random seed generator used in initialization (int, optional)
        - **representation**: Specifies whether to use the last linear layer embeddings or the raw data. Must be one of 'linear' or 'raw' (string, optional)
        - **kmeans_args**: Specifies additional kmeans-related parameters
        
            - **tol**: Specifies the value of the Frobenius norm of the inertia tensor by which kmeans should cease (float, optional)
            - **max_iter**: Specifies the maximum number of iterations that kmeans should use before terminating (int, optional)
            - **n_init**: Specifies the number of kmeans run-throughs to use, wherein the one with the smallest inertia is selected for the selection phase (int, optional)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}): 
        
        super(KMeansSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
  
        if 'rand_seed' in args:
            random.seed(args['rand_seed'])
            
        if 'representation' in args:
            self.representation = args['representation']
        else:
            self.representation = 'linear'
            
        if 'kmeans_args' in args:
            self.kmeans_args = args['kmeans_args']
        else:
            args['kmeans_args'] = dict()
            self.kmeans_args = args['kmeans_args']
            
        if 'tol' not in self.kmeans_args:
            self.kmeans_args['tol'] = 1e-4
            
        if 'max_iter' not in self.kmeans_args:
            self.kmeans_args['max_iter'] = 300

        if 'n_init' not in self.kmeans_args:
            self.kmeans_args['n_init'] = 10
    
    def get_closest_distances(self, ground_set, center_tensor):
        
        ground_set_loader = DataLoader(ground_set, batch_size = self.args['batch_size'], shuffle = False)

        # Store the minimum distances in this tensor    
        ground_set_min_distances = torch.zeros(len(ground_set)).to(self.device)
        ground_set_closest_center_indices = torch.zeros(len(ground_set), dtype=torch.long).to(self.device)
        evaluated_ground_set_points = 0
    
        with torch.no_grad():
            for ground_set_batch_idx, ground_set_batch in enumerate(ground_set_loader):
                
                # Put the batch on the correct device, calculate embedding, initialize this batch's min distance center to None
                if self.representation == 'linear':
                    ground_set_batch = CustomTensorDataset(ground_set_batch.to(self.device))
                    ground_set_batch = self.get_embedding(ground_set_batch)
                elif self.representation == 'raw':
                    ground_set_batch = ground_set_batch.to(self.device).view(len(ground_set_batch), -1)
                else:
                    raise ValueError("Representation must be one of 'linear', 'raw'")
                                 
                # Calculate the distance of each point in the ground set batch to each center in the center batch.
                inter_batch_distances = torch.cdist(ground_set_batch, center_tensor, p=2)
                    
                # Calculate the minimum distances across each row; this will reflect the distance to the closest center
                batch_min_distances, batch_min_idx = torch.min(inter_batch_distances, dim=1)                   
                    
                # Assign minimum distance to the correct slice of the storage tensor
                ground_set_min_distances[evaluated_ground_set_points:(evaluated_ground_set_points + len(ground_set_batch))] = batch_min_distances
                ground_set_closest_center_indices[evaluated_ground_set_points:(evaluated_ground_set_points + len(ground_set_batch))] = batch_min_idx
                evaluated_ground_set_points += len(ground_set_batch)

        return ground_set_min_distances, ground_set_closest_center_indices.tolist()
    
    def kmeans_plusplus(self, num_centers):
        
        # 1. Choose a random point to be the center (uniform dist)
        selected_points = [random.choice(len(self.unlabeled_dataset))]
        
        # Keep repeating this step until num_centers centers have been chosen
        while len(selected_points) < num_centers:
            
            # 2. Calculate the squared distance to the nearest center for each point
            selected_centers = Subset(self.unlabeled_dataset, selected_points)
            if self.representation == 'linear':
                selected_centers_tensor = self.get_embedding(selected_centers)
            elif self.representation == 'raw':
                selected_centers_tensor = next(iter(DataLoader(selected_centers, shuffle=False, batch_size=len(selected_points)))).to(self.device)
                selected_centers_tensor = selected_centers_tensor.view(len(selected_points), -1)
            else:
                raise ValueError("Representation must be one of 'linear', 'raw'")
            
            ground_set_min_distances, _ = self.get_closest_distances(self.unlabeled_dataset, selected_centers_tensor)
            ground_set_min_distances = torch.pow(ground_set_min_distances, 2)
            
            # 3. Sample a random point with probability proportional to the squared distance
            # Note: torch.multinomial does not require that the weight tensor sum to 1 
            # (e.g., forms a probability distribution). It simply requires non-negative weights 
            # and will form the distribution itself. torch.multinomial can be used as it allows 
            # the tensor to stay on the GPU and because sampling from the multinomial distribution 
            # assigns the probability of sampling element i with the calculated distance weight. 
            distance_probability_distribution = ground_set_min_distances
            random_choice = torch.multinomial(distance_probability_distribution, 1).item()
            
            # 4. Add the chosen index to the center list
            selected_points.append(random_choice)
            
        return selected_points

    def kmeans_calculate_means(self, clusters):

        # Calculate dimensions of and form the storage tensor        
        if self.representation == 'linear':
            num_features = self.model.get_embedding_dim()
            means = torch.zeros(len(clusters), num_features).to(self.device)     
        elif self.representation == 'raw':
            num_features = self.unlabeled_dataset[0].view(-1).shape[0]
            means = torch.zeros(len(clusters), num_features).to(self.device)
        else:
            raise ValueError("Representation must be one of 'linear', 'raw'")
            
        # Calculate the mean of each cluster
        for i, cluster in enumerate(clusters):
            
            # Only load those points specific to the cluster
            ground_set_cluster = Subset(self.unlabeled_dataset, cluster)
            ground_set_loader = DataLoader(ground_set_cluster, batch_size = self.args['batch_size'], shuffle = False)
    
            running_cluster_sum = None
    
            # Calculate the sum of all the elements in the cluster after getting the correct embedding
            for batch_idx, ground_set_batch in enumerate(ground_set_loader):
                
                # Put the center batch on the correct device, calculate embedding
                if self.representation == 'linear':
                    ground_set_batch = CustomTensorDataset(ground_set_batch.to(self.device))
                    ground_set_batch = self.get_embedding(ground_set_batch)
                elif self.representation == 'raw':
                    ground_set_batch = ground_set_batch.to(self.device).view(len(ground_set_batch), -1)    
                
                if running_cluster_sum is None:
                    running_cluster_sum = torch.sum(ground_set_batch, dim=0)
                else:
                    running_cluster_sum += torch.sum(ground_set_batch, dim=0)
         
            # Divide by total number of elements to get the mean
            running_cluster_sum /= len(ground_set_cluster)
            means[i] = running_cluster_sum

        return means
    
    def kmeans_calculate_clusters(self, center_tensor):
        
        # Calculate the closest center indices
        _, ground_set_closest_center_indices = self.get_closest_distances(self.unlabeled_dataset, center_tensor)
        
        # For each center, create an associated cluster and add points to them
        clusters = [[] for x in range(len(center_tensor))]
        for i, index in enumerate(ground_set_closest_center_indices):
            clusters[index].append(i)
            
        # Return the clusters
        return clusters
        

    def kmeans_clustering(self, num_centers):

        best_inertia = None
        best_centers = None        

        # Run kmeans algorithm n_init times and choose the one with best inertia.
        for i in range(self.kmeans_args['n_init']):   
            
            # Use kmeans++ initialization
            centers_subset = Subset(self.unlabeled_dataset, self.kmeans_plusplus(num_centers))
            
            if self.representation == "linear":
                centers = self.get_embedding(centers_subset)
            else:
                centers = next(iter(DataLoader(centers_subset, shuffle=False, batch_size=len(centers_subset)))).to(self.device)
        
            # Alternate between means/assignment steps until max_iter reached
            for i in range(self.kmeans_args['max_iter']):
                old_centers = centers
                clusters = self.kmeans_calculate_clusters(centers)
                centers = self.kmeans_calculate_means(clusters)
            
                center_diff = centers - old_centers
                frobenius_norm = torch.linalg.norm(center_diff, ord="fro").item()
            
                # If frobenius norm of difference between centers is below a tolerance, stop this kmeans iteration.
                if frobenius_norm < self.kmeans_args['tol']:
                    break

            # Lastly, evaluate the inertia of this kmeans solution. If it is the best one so far, keep it.
            ground_set_min_distances, ground_set_closest_center_indices = self.get_closest_distances(self.unlabeled_dataset, centers)
            inertia = torch.pow(ground_set_min_distances, 2).sum().item()

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers

        return best_centers

    def select(self, budget):
        
        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	
        
        self.model.eval()
        
        # Get the best centers through kmeans clustering
        best_centers = self.kmeans_clustering(budget)
        
        # Choose the point closest to each center.
        ground_set_min_distances, ground_set_closest_center_indices = self.get_closest_distances(self.unlabeled_dataset, best_centers)
        
        cluster_min_distances = [None for x in range(budget)]
        cluster_min_indices = [None for x in range(budget)]
        
        for i, (distance, center_index) in enumerate(zip(ground_set_min_distances, ground_set_closest_center_indices)):            
            if cluster_min_distances[center_index] is None or distance < cluster_min_distances[center_index]:
                cluster_min_distances[center_index] = distance
                cluster_min_indices[center_index] = i

        # Return the list of indices of points that are closest to the best centers chosen by kmeans
        return cluster_min_indices                