import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class MnistNetwork(nn.Module):

    def __init__(self, input_size=784, hidden_dims=(128), num_classes=10):
        super(MnistNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.build_network()

    def build_network(self):
        all_dims = [self.input_size] + self.hidden_dims + [self.num_classes]
        all_layers = []
        for i in range(len(all_dims)-1):
            all_layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            if i != len(all_dims) - 2:
                all_layers.append(nn.ReLU())
        self.model = nn.Sequential(*all_layers)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)




# TODO: Load and preprocess the MNIST dataset

# TODO: Initialize the network, loss function, and optimizer

# TODO: Implement the training loop

# TODO: Evaluate the network's accuracy on the test set

