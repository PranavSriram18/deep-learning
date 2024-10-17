import torch

def can_broadcast(dims0, dims1) -> bool:
    for dim0, dim1 in zip(reversed(dims0), reversed(dims1)):
        if dim0 != dim1 and dim0 != 1 and dim1 != 1:
            return False
    return True

def all_pairwise_dists(X):
    """
    Given n x d matrix X, representing n points in a d-dimensional space.
    Return a matrix D, whose (i, j) entry is the Euclidean distance between
    the ith and jth points.

    Solution:
    Expand expression for squared dist between X_i, X_j. It works out as
    r_i + r_j - 2<X_i, X_j>
    where r is the vector of squared norms of the rows of X.
    """
    r = torch.sum(X ** 2, dim=1, keepdim=True)  # Nxd -> Nx1
    Y = X @ X.t() # N x N
    return (r + r.t() - 2 * Y) ** 0.5