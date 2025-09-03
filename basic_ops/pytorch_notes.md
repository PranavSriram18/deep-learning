
### Cat vs Stack
torch.cat concatenates along an existing dim (default dim 0, can specify with dim=)
torch.stack inserts a new dim at the front

e.g. 
cat'ing 2 (3, 2) tensors gives a (6, 2) tensor
stacking 2 (3, 2) tensors gives a (2, 3, 2) tensor

stack is useful for adding a batch dimension


### Advanced indexing

# 1. Basic case - single dimension indexing
x = torch.arange(5)
idx = torch.tensor([0, 2, 4])
result = x[idx]  # Gets elements at positions 0, 2, 4
print("1. Basic indexing:\n", result)  # tensor([0, 2, 4])

# 2. Two dimensions - separate indices for each dim
x = torch.arange(12).reshape(4, 3)
# tensor([[ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 9, 10, 11]])
row_idx = torch.tensor([0, 2])
col_idx = torch.tensor([1, 2])
result = x[row_idx, col_idx]  # Gets elements at (0,1) and (2,2)
print("\n2. Two dim indexing:\n", result)  # tensor([1, 8])

# 3. Broadcasting with multiple indices
x = torch.arange(12).reshape(4, 3)
row_idx = torch.tensor([0, 2]).unsqueeze(1)  # Shape: (2, 1)
col_idx = torch.tensor([1, 2]).unsqueeze(0)  # Shape: (1, 2)
result = x[row_idx, col_idx]  # Gets all combinations
print("\n3. Broadcast indexing:\n", result)
# Shape: (2, 2)
# Gets elements at: (0,1), (0,2), (2,1), (2,2)

# 4. Batched indexing
B, N = 2, 3
x = torch.arange(B*N*N).reshape(B, N, N)
batch_idx = torch.arange(B).unsqueeze(-1)  # Shape: (B, 1)
indices = torch.tensor([[0, 2],
                       [1, 2]])  # Shape: (B, 2)
result = x[batch_idx, indices]  # Select rows for each batch
print("\n4. Batched indexing:\n", result)

# 5. Key rules about shapes:
# - When multiple index tensors are provided, they are broadcast together
# - The result shape includes:
#   a) All broadcast dimensions from the index tensors
#   b) Any remaining dimensions from the input tensor

# 6. Common patterns:
x = torch.randn(10, 20, 30)  # Example tensor

# Select specific entries:
idx1 = torch.tensor([0, 2, 4])
idx2 = torch.tensor([1, 1, 1])
idx3 = torch.tensor([5, 6, 7])
result = x[idx1, idx2, idx3]  # Shape: (3,)

# Select same indices across multiple dims:
idx = torch.tensor([0, 2, 4])
result = x[idx, idx, idx]  # Shape: (3,)

# Select all combinations (outer product):
idx1 = torch.tensor([0, 2]).unsqueeze(1)  # (2, 1)
idx2 = torch.tensor([1, 3, 5]).unsqueeze(0)  # (1, 3)
result = x[idx1, idx2]  # Shape: (2, 3, 30)


### Boolean Mask Indexing
x[mask] always yields a 1D flattened tensor
This is a gotcha but is also super useful sometimes


### Misc notes
Note that we need .item() to get the scalar out of a shape (1,) tensor
(e.g. the result of calling torch.mean() on a tensor)


### Training code
def train(self, lr: float, batch_size: int, steps: int, print_every: int) -> None:
    
    # 1. create optimizer from model.parameters() and lr 
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    for i in range(steps):
        xb, yb = self.loader.get_batch('train')  # BxC, BxC
        
        # 2. evaluate the loss by using model as a callable
        logits, loss = self.model(xb, yb)
        if (i % print_every == 0):
            print(f"Logits on sample 0: {logits[0]}")
            # 3. use .item() to extract scalar from loss tensor
            self.print_sample(i, loss.item())

        # 4. zero the grad, run backwards, step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()