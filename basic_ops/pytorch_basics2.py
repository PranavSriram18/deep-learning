import torch 

def exercise1() -> float:
    """ 
    Create two tensors:

    A 3x3 tensor filled with random numbers from a normal distribution.
    A 3x3 tensor filled with ones.

    Then, perform the following operations:
    a) Add these two tensors together.
    b) Multiply the result elementwise by 2.
    c) Compute the mean of all elements in the resulting tensor.
    """
    x = torch.randn(size=(3, 3))
    y = torch.ones(size=(3, 3))
    z = (x + y) * 2
    return torch.mean(z).item()

def exercise2() -> float:
    """
    Tensor Reshaping, Slicing, and Operations

    Create a 1D tensor with the values [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].
    Reshape this tensor into a 3x4 matrix.
    Slice this matrix to get a 2x3 submatrix containing elements from the first
    two rows and first three columns.
    Transpose this 2x3 submatrix.
    Multiply this transposed matrix element-wise with a new 3x2 matrix filled
    with values from a uniform distribution between 0 and 1.
    Sum up all elements in the resulting matrix.
    """
    x = torch.tensor([i for i in range(1, 13)])
    y = x.reshape(3, 4)
    z = y[0:2, 0:3]
    zt = z.transpose()  # 3x2
    zt *= torch.rand(3, 2)
    return torch.sum(zt).item()

def exercise3() -> float:
    """
    Exercise 3: Advanced Tensor Manipulation

    Create a 4D tensor of shape (2, 3, 4, 5) filled with random values from a 
    normal distribution.
    Create a 2D tensor of shape (3, 4) filled with random ints between 0 and 4.
    Use the 2D tensor to index into the last two dimensions of the 4D tensor.
    Compute the mean along the second dimension (dim=1) of the resulting tensor.
    Create a 2D tensor of shape (2, 5) from a uniform dist over [-1, 1].
    Perform a matmul between the result from step 4 and the tensor from step 5.
    Apply a softmax function along the last dimension of the result.
    Compute the sum of the logarithm of this result.
    """
    x = torch.rand(2, 3, 4, 5)
    idxs = torch.randint(low=0, high=5, size=(3, 4))

