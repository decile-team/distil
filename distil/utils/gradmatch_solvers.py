import torch
from scipy.optimize import nnls

def Fixed_Weight_Greedy_Parallel(A, b, val_set_size, nnz=None, device="cpu"):
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
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)

    # Calculate b * n / val_set_size
    b_k_val = (n / val_set_size) * b

    # Stores memoized version of Ax, where x is the vector containing an element from {0,1}
    memoized_Ax = torch.zeros(d, device = device)

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

        """
        if gain_norms[actual_index] < resid_norm:
            x[actual_index] = 1
            memoized_Ax = memoized_Ax + A[:,actual_index]
        else:
            break
        """

    return x

# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_Parallel(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
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
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
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
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)

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