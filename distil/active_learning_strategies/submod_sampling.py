from .strategy import Strategy

import submodlib

class SubmodularSampling(Strategy):
    
    """
    This strategy uses one of the submodular functions viz. 'facility_location', 'feature_based', 'graph_cut', 
    'log_determinant', 'disparity_min', or 'disparity_sum' :footcite:`iyer2021submodular`, :footcite:`dasgupta-etal-2013-summarization`
    to select new points via submodular maximization. These techniques can be applied directly to the features/embeddings 
    or on the gradients of the loss functions.
    
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
        - **submod_args**: Additional parameters for submodular selection (dict, optional)
        
            - **submod**: The choice of submodular function to use. Must be one of 'facility_location', 'feature_based', 'graph_cut', 'log_determinant', 'disparity_min', 'disparity_sum' (string)
            - **metric**: The similarity metric to use in relevant functions. Must be one of 'cosine' or 'euclidean' (string)
            - **representation**: The representation of each data point to be used in submodular selection. Must be one of 'linear', 'grad_bias', 'grad_linear', 'grad_bias_linear' (string)
            - **feature_weights**: If using 'feature_based', then this specifies the weights for each feature (list)
            - **concave_function**: If using 'feature_based', then this specifies the concave function to apply in the feature-based objective (typing.Callable)
            - **lambda_val**: If using 'graph_cut' or 'log_determinant', then this specifies the lambda constant to be used in both functions (float)
            - **optimizer**: The choice of submodular optimization technique to use. Must be one of 'NaiveGreedy', 'StochasticGreedy', 'LazyGreedy', or 'LazierThanLazyGreedy' (string)
            - **stopIfZeroGain**: Whether to stop if adding a point results in zero gain in the submodular objective function (bool)
            - **stopIfNegativeGain**: Whether to stop if adding a point results in negative gain in the submodular objective function (bool)
            - **verbose**: Whether to print more verbose output
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(SubmodularSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'submod_args' in args:
            self.submod_args = args['submod_args']
        else:
            self.submod_args = {'submod': 'facility_location',
                                'metric': 'cosine',
                                'representation': 'linear'}
            
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
        
        # Get the ground set size, which is the size of the unlabeled dataset
        ground_set_size = len(self.unlabeled_dataset)
        
        # Get the representation of each element.
        if 'representation' in self.submod_args:
            representation = self.submod_args['representation']
        else:
            representation = 'linear'
        
        if representation == 'linear':
            ground_set_representation = self.get_embedding(self.unlabeled_dataset)
        elif representation == 'grad_bias':
            ground_set_representation = self.get_grad_embedding(self.unlabeled_dataset, True, "bias")
        elif representation == 'grad_linear':
            ground_set_representation = self.get_grad_embedding(self.unlabeled_dataset, True, "linear")
        elif representation == 'grad_bias_linear':
            ground_set_representation = self.get_grad_embedding(self.unlabeled_dataset, True, "bias_linear")
        else:
            raise ValueError("Provided representation must be one of 'linear', 'grad_bias', 'grad_linear', 'grad_bias_linear'")            
        
        if self.submod_args['submod'] == 'facility_location':
            if 'metric' in self.submod_args:
                metric = self.submod_args['metric']
            else:
                metric = 'cosine'
            submod_function = submodlib.FacilityLocationFunction(n=ground_set_size,
                                                                 mode="dense",
                                                                 data=ground_set_representation.cpu().numpy(),
                                                                 metric=metric)
        elif self.submod_args['submod'] == "feature_based":
            if 'feature_weights' in self.submod_args:
                feature_weights = self.submod_args['feature_weights']
            else:
                feature_weights = None
                
            if 'concave_function' in self.submod_args:
                concave_function = self.submod_args['concave_function']
            else:
                from submodlib_cpp import FeatureBased
                concave_function = FeatureBased.logarithmic
                
            submod_function = submodlib.FeatureBasedFunction(n=ground_set_size,
                                                             features=ground_set_representation.cpu().numpy().tolist(),
                                                             numFeatures=ground_set_representation.shape[1],
                                                             sparse=False,
                                                             featureWeights=feature_weights,
                                                             mode=concave_function)
        elif self.submod_args['submod'] == "graph_cut":
            if 'lambda_val' not in self.submod_args:
                raise ValueError("Graph Cut Requires submod_args parameter 'lambda_val'")
            
            if 'metric' in self.submod_args:
                metric = self.submod_args['metric']
            else:
                metric = 'cosine'
            
            submod_function = submodlib.GraphCutFunction(n=ground_set_size,
                                                         mode="dense",
                                                         lambdaVal=self.submod_args['lambda_val'],
                                                         data=ground_set_representation.cpu().numpy(),
                                                         metric=metric)
        elif self.submod_args['submod'] == 'log_determinant':
            if 'lambda_val' not in self.submod_args:
                raise ValueError("Log Determinant Requires submod_args parameter 'lambda_val'")
            
            if 'metric' in self.submod_args:
                metric = self.submod_args['metric']
            else:
                metric = 'cosine'
            
            submod_function = submodlib.LogDeterminantFunction(n=ground_set_size,
                                                         mode="dense",
                                                         lambdaVal=self.submod_args['lambda_val'],
                                                         data=ground_set_representation.cpu().numpy(),
                                                         metric=metric)
        elif self.submod_args['submod'] == 'disparity_min':
            if 'metric' in self.submod_args:
                metric = self.submod_args['metric']
            else:
                metric = 'cosine'
            submod_function = submodlib.DisparityMinFunction(n=ground_set_size,
                                                             mode="dense",
                                                             data=ground_set_representation.cpu().numpy(),
                                                             metric=metric)
        elif self.submod_args['submod'] == 'disparity_sum':
            if 'metric' in self.submod_args:
                metric = self.submod_args['metric']
            else:
                metric = 'cosine'
            submod_function = submodlib.DisparitySumFunction(n=ground_set_size,
                                                             mode="dense",
                                                             data=ground_set_representation.cpu().numpy(),
                                                             metric=metric)
        else:
            raise ValueError(F"{self.submod_args['submod']} is not currently supported. Choose one of 'facility_location', 'feature_based', 'graph_cut', 'log_determinant', 'disparity_min', or 'disparity_sum'")
            
        # Get solver arguments
        optimizer = self.submod_args['optimizer'] if 'optimizer' in self.submod_args else 'LazyGreedy'
        stopIfZeroGain = self.submod_args['stopIfZeroGain'] if 'stopIfZeroGain' in self.submod_args else False
        stopIfNegativeGain = self.submod_args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.submod_args else False
        verbose = self.submod_args['verbose'] if 'verbose' in self.submod_args else False
        
        # Use solver to get indices from the filtered set via the submodular function
        greedy_list = submod_function.maximize(budget=budget,
                                              optimizer=optimizer,
                                              stopIfZeroGain=stopIfZeroGain,
                                              stopIfNegativeGain=stopIfNegativeGain,
                                              verbose=verbose)
        greedy_indices = [x[0] for x in greedy_list]
        return greedy_indices