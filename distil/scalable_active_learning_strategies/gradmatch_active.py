import numpy as np
import torch

from .strategy import Strategy
from scipy.optimize import nnls

class GradMatchActive(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}, validation_dataset = None):
        
        # Run super constructor
        super(GradMatchActive, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'grad_embedding' not in args:
            self.grad_embedding = 'bias_linear'
        else:
            self.grad_embedding = args['grad_embedding']
        
        if 'omp_reg' not in args:
            self.omp_reg = 0
        else:
            self.omp_reg = args['omp_reg']
        
        self.validation_dataset = validation_dataset
            
    def fixed_weight_greedy_parallel(self, A, b, val_set_size, nnz=None):
        '''approximately solves min_x ||Ax - b||_2 s.t. x_i \in \{0,1\}
        Args:
        A: design matrix of size (d, n)
        b: measurement vector of length d
        nnz = maximum number of nonzero coefficients (if None set to n)
        Returns:
        vector of length n
        '''
        d, n = A.shape
        if nnz is None:
            nnz = n
        x = torch.zeros(n, device=self.device)  # ,dtype=torch.float64)

        # Calculate b * n / val_set_size
        b_k_val = (n / val_set_size) * b

        # Stores memoized version of Ax, where x is the vector containing an element from {0,1}
        memoized_Ax = torch.zeros(d, device=self.device)

        for i in range(nnz):
        
            # Calculate residual
            resid = memoized_Ax - b_k_val

            # Calculate columnwise difference with resid.
            # resid[:None] promotes shape from (d) to (d,)
            gain_norms = (A + resid[:,None]).norm(dim=0)

            # Choose smallest norm from among columns of A not already chosen.
            # Such a vector represents the best greedy choice in matching to 
            # the residual.
            zero_x = torch.nonzero(x == 0)
            argmin = torch.argmin(gain_norms[zero_x])
            actual_index = zero_x[argmin].item()

            # Add this column to the memoized Ax. Set this weight to 1.
            x[actual_index] = 1
            memoized_Ax = memoized_Ax + A[:,actual_index]

            # One condition for stopping is to check if the gain_norm is larger than the residual norm.
            # However, we opt to select the full number of points specified while following this 
            # greedy behavior in order to fill the AL budget.
            """
            if gain_norms[actual_index] < resid_norm:
                x[actual_index] = 1
                memoized_Ax = memoized_Ax + A[:,actual_index]
            else:
                break
            """

        return x

    # NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
    def orthogonalmp_reg_parallel(self, A, b, tol=1E-4, nnz=None, positive=False, lam=1):
        '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
        Args:
            A: design matrix of size (d, n)
            b: measurement vector of length d
            tol: solver tolerance
            nnz = maximum number of nonzero coefficients (if None set to n)
            positive: only allow positive nonzero coefficients
        Returns:
            vector of length n
        '''
        AT = torch.transpose(A, 0, 1)
        d, n = A.shape
        if nnz is None:
            nnz = n
        x = torch.zeros(n, device=self.device)  # ,dtype=torch.float64)
        resid = b.detach().clone()
        normb = b.norm().item()
        indices = []

        for i in range(nnz):
            if resid.norm().item() / normb < tol:
                break
            projections = torch.matmul(AT, resid)

            if positive:
                index = torch.argmax(projections)
            else:
                index = torch.argmax(torch.abs(projections))

            if index in indices:
                break

            indices.append(index)
            if len(indices) == 1:
                A_i = A[:, index]
                x_i = projections[index] / torch.dot(A_i, A_i).view(-1)
                A_i = A[:, index].view(1, -1)
            else:
                A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)
                temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=self.device)

                # If constrained to be positive, use nnls. Otherwise, use torch's lstsq method.
                if positive:
                    x_i, _ = nnls(temp.cpu().numpy(), torch.matmul(A_i, b).view(-1, 1).cpu().numpy()[:,0])
                    x_i = torch.Tensor(x_i).cuda()
                else:
                    x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)

            resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)

        x_i = x_i.view(-1)
        for i, index in enumerate(indices):
            try:
                x[index] += x_i[i]
            except IndexError:
                x[index] += x_i

        return x    
    
    def select(self, budget, use_weights=False):
        
        """
        Select next set of points
        
        Parameters
        ----------
        budget: int
            Number of indexes to be returned for next set
        use_weights: bool
            Whether to use fixed-weight version (false) or OMP version (true)
        
        Returns
        ----------
        subset_idxs: list
            List of selected data point indexes with respect to unlabeled_x and, if use_weights is true, the weights associated with each point
        """ 
        
        # Compute hypothesized gradients.
        hypothesized_gradients = self.get_grad_embedding(self.unlabeled_dataset, True, grad_embedding_type=self.grad_embedding)
        
        # If there is a validation set, set the target gradient vector to be the average validation gradient.
        # Otherwise, set the target gradient vector to be the average hypothesized gradient
        if self.validation_dataset is not None:
            target_gradient = self.get_grad_embedding(self.validation_dataset, False, grad_embedding_type=self.grad_embedding).sum(dim=0)
            num_target_gradient_summed_vectors = len(self.validation_dataset)
        else:
            target_gradient = hypothesized_gradients.sum(dim=0)
            num_target_gradient_summed_vectors = len(self.unlabeled_dataset)
            
        # Needed for solvers
        transposed_hypothesized_gradients = torch.transpose(hypothesized_gradients, 0, 1)
            
        # Use different solvers depending on the use of weights
        if use_weights:
            
            # We solve the weighted objective; hence, the OMP solver should be used.
            weight_vector = self.orthogonalmp_reg_parallel(transposed_hypothesized_gradients, target_gradient, nnz=budget, positive=True, lam=self.omp_reg)
            non_negative_weight_indices = torch.nonzero(weight_vector).view(-1)
            non_negative_weights = weight_vector[non_negative_weight_indices].tolist()
            non_negative_weight_indices = non_negative_weight_indices.tolist()
            
            # OMP solver might not return 'budget' number of non-negative weights. Pick the rest randomly and assign weights of 1. 
            diff = budget - len(non_negative_weight_indices)

            if diff > 0:
                remain_list = set(np.arange(len(self.unlabeled_dataset))).difference(set(non_negative_weight_indices))
                new_idxs = np.random.choice(list(remain_list), size=diff, replace=False)
                non_negative_weight_indices.extend(new_idxs)
                non_negative_weights.extend([1 for _ in range(diff)])
                
            idxs = non_negative_weight_indices
            gammas = torch.tensor(non_negative_weights).to(self.device)
            
            return idxs, gammas
        
        else:
            
            # We solve the fixed weight objective
            weight_vector = self.fixed_weight_greedy_parallel(transposed_hypothesized_gradients, target_gradient, num_target_gradient_summed_vectors, nnz=budget)
            non_negative_weight_indices = torch.nonzero(weight_vector).view(-1).tolist()
            
            # Unlike the other wrapper, this solver will ensure there are 'budget' non-negative weights.
            return non_negative_weight_indices