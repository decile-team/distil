from .strategy import Strategy

import submodlib

class SubmodularSampling(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(SubmodularSampling, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'submod_args' in args:
            self.submod_args = args['submod_args']
        else:
            self.submod_args = {'submod': 'facility_location',
                                'metric': 'cosine',
                                'representation': 'linear'}
            
    def select(self, budget):
        
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
                                                                 data=ground_set_representation.numpy(),
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
                                                             features=ground_set_representation.numpy().tolist(),
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
                                                         data=ground_set_representation.numpy(),
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
                                                         data=ground_set_representation.numpy(),
                                                         metric=metric)
        elif self.submod_args['submod'] == 'disparity_min':
            if 'metric' in self.submod_args:
                metric = self.submod_args['metric']
            else:
                metric = 'cosine'
            submod_function = submodlib.DisparityMinFunction(n=ground_set_size,
                                                             mode="dense",
                                                             data=ground_set_representation.numpy(),
                                                             metric=metric)
        elif self.submod_args['submod'] == 'disparity_min':
            if 'metric' in self.submod_args:
                metric = self.submod_args['metric']
            else:
                metric = 'cosine'
            submod_function = submodlib.DisparitySumFunction(n=ground_set_size,
                                                             mode="dense",
                                                             data=ground_set_representation.numpy(),
                                                             metric=metric)
        else:
            raise ValueError(F"{self.submod_args['submod']} is not currently supported. Choose one of 'facility_location', 'feature_based', 'graph_cut', 'log_determinant', 'disparity_min', or 'disparity_sum'")
            
        # Get solver arguments
        optimizer = self.args['optimizer'] if 'optimizer' in self.args else 'NaiveGreedy'
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