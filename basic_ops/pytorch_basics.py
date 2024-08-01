import torch

class Solution:
    def exercise0():
        """ 
        Exercise:
        Create a PyTorch tensor representing a 4x4 matrix of random integers 
        between 1 and 10 (inclusive) 
        Then, perform the following operations:

        Extract the upper-right 2x2 submatrix.
        Compute the sum of this submatrix.
        Create a new 2x2 matrix where each element is the original submatrix 
        element divided by the sum.
        Concatenate this new matrix horizontally with a 2x2 identity matrix.
        Print the result, showing the data type and device of the final tensor.
        """

        x = torch.randint(low=1, high=11, size=(4, 4), dtype=torch.float)

        upper_right = x[0:2, 2:]
        usum = upper_right.sum()
        upper_right_avg = upper_right / usum

        id = torch.eye(2, dtype=torch.int)
        res = torch.hstack([upper_right_avg, id])
        print(f"Result: {res} \n device: {res.device} \n dtype: {res.dtype}")

    def exercise1():
        """
        Exercise:
        Create a PyTorch tensor representing a 5x5 matrix of random 
        floating-point numbers between 0 and 1. Then, perform the following operations:

        Apply a mask to the tensor, setting all elements less than 0.5 to 0.
        Compute the row-wise mean of the resulting tensor.
        Create a new tensor by subtracting these means from each row of the 
        original masked tensor.
        Find the maximum value in each column of this new tensor.
        Create a 1D tensor containing the indices of these maximum values.
        Print the original tensor, the masked tensor, the mean-subtracted 
        tensor, and the final result.
        """

        x = torch.rand((5, 5), dtype=torch.float)
        x_mask = x.where(x >= 0.5, x, torch.zeros_like(x))  # can also do x * (x >= 0.5)
        row_means = x_mask.mean(dim=1, keepdim=True)  # 5 x 1
        y = x_mask - row_means  # 5 x 5
        _, column_max_idxs = torch.max(y, dim=0, keepdim=False)  # (5), (5)
        torch.set_printoptions(precision=4) 
        print(f"original: {x} \n masked: {x_mask} \n mean-subtracted: {y} \n max_indices: {column_max_idxs}")

    def exercise2():
        """ 
        Exercise:
        Create a PyTorch tensor representing a batch of 10 grayscale images, 
        each 28x28 pixels (similar to MNIST data). Then, perform the following operations:

        Normalize the pixel values to be between 0 and 1.
        Add random noise to the images (Gaussian noise with mean 0 and standard deviation 0.1).
        Apply a 3x3 average pooling operation to the noisy images.
        Create a simple neural network with one hidden layer (input: 784, hidden: 128, output: 10).
        Pass the pooled images through this network.
        Compute the softmax of the output.
        Print the shapes of the tensor right before the network and the final 
        probabilities for the first image.
        """

        batch_sz = 10
        hidden_dim = 128
        num_classes = 10

        x = torch.randint(low=0, high=256, size=(batch_sz, 28, 28), dtype=torch.float)
        x /= 255.0  # normalize pixel values to [0, 1]

        noise = torch.randn_like(x) / 10.  # divide by 10 so stddev is 0.1
        x += noise
        x = torch.clamp(x, 0, 1)
        pooler = torch.nn.AvgPool2d(kernel_size=(3,3))
        x = pooler(x)
        x = torch.reshape(x, shape=(batch_sz, -1))
        _, d = x.shape

        print(f"Shape of input to network: {x.shape}")

        w1 = torch.randn(size=(d, hidden_dim))
        w2 = torch.randn(size=(hidden_dim, num_classes))

        h1 = x @ w1  # (Bxd) @ (dxh) -> (Bxh)
        h1_r = torch.relu(h1)
        logits = h1_r @ w2  # (Bxh) @ (hxC) -> (BxC)
        probs = torch.softmax(logits, dim=1)  # (BxC) -> (BxC)

        print(f"Probs for 1st image: {probs[0, :]}")

    def exercise3():
        """
        Exercise: Temperature Conversion
        Create a PyTorch tensor representing daily maximum temperatures in Celsius
        for a week. Then, perform the following operations:

        Create a 1D tensor with 7 random integers between 20 and 35 (
        representing Celsius temperatures).
        Reshape this tensor into a 7x1 column vector.
        Convert the temperatures from Celsius to Fahrenheit using broadcasting.
        Calculate the average temperature for the week in Fahrenheit. 
        """
        temps = torch.randint(low=20, high=36, size=(7,))
        temps = temps.reshape((7,1))
        temps_fahr = temps * 1.8 + 32.
        print("Temperatures in Celsius:\n", temps)
        print("Temperatures in Fahrenheit:\n", temps_fahr)
        print(f"Avg temp in fahrenheit: {torch.mean(temps_fahr)}")

 










