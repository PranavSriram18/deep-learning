import torch

class LinearLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = torch.randn(input_size, output_size)
        self.biases = torch.randn(output_size)
        
    def forward(self, inputs):
        # (B x d_in) (d_in x d_out) -> B x d_out
        out = torch.matmul(inputs, self.weights) + self.biases  
        self.cache = inputs
        return out
        
    def backward(self, grad_outputs):
        """
        Implements one step of the backward pass.
        """
        # grad_outputs is B x d_out
        # weights is d_in x d_out
        # inputs (self.cache) is B x d_in

        # (d_in x B) @ (B x d_out) -> d_in x d_out
        grad_weights = torch.matmul(torch.transpose(self.cache), grad_outputs) 
        # (B x d_out) @ (d_out x d_in) -> B x d_in
        grad_inputs = torch.matmul(grad_outputs, torch.transpose(self.weights))
        # (B x d_out) -> (d_out,)
        grad_biases = torch.sum(grad_outputs, dim=0, keepdim=False)  
        
        return grad_inputs, grad_weights, grad_biases
    
class Optimizer:
    def step(self, layer, inputs, targets, lr=0.01):
        outputs = layer.forward(inputs)  # B x d_out
        loss = torch.mean((targets - outputs) ** 2) / 2.  # half MSE
        grad_outputs = outputs - targets  # B x d_out
        B, d_out = grad_outputs.shape
        grad_outputs /= (B * d_out)
        grad_inputs, grad_weights, grad_biases = layer.backward(grad_outputs)
        layer.weights -= lr * grad_weights
        layer.biases -= lr * grad_biases

        print(f"Loss: {loss.item()}")
        print(f"Grad Inputs Shape: {grad_inputs.shape}")
        print(f"Grad Weights Shape: {grad_weights.shape}")
        print(f"Grad Biases Shape: {grad_biases.shape}")
    
def run(batch_size, input_size, output_size, steps, lr):
    inputs = torch.randn((batch_size, input_size))
    targets = torch.randn((batch_size, output_size))
    layer = LinearLayer(input_size, output_size)
    optim = Optimizer()
    for i in range(steps):
        optim.step(layer, inputs, targets, lr)

if __name__=="__main__":
    run(32, 32, 8, steps=10, lr=0.01)
