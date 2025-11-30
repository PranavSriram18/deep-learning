
from layers.layer_config import AttentionConfig, MLPConfig

import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import functional as F # type: ignore

class AttentionHead(nn.Module):
    """
    One head of ordinary causal self-attention.
    H: head size 
    D: embedding dimension of X
    C: context length (block size)
    """
    def __init__(self, H: int, D: int, C: int):
        super().__init__()
        self.key = nn.Linear(D, H, bias=False)
        self.query = nn.Linear(D, H, bias=False)
        self.value = nn.Linear(D, H, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(C, C)))

    def forward(self, X: torch.tensor) -> torch.tensor:
        # Linear projections here are all (B, C, D) -> (B, C, H)
        # Explanatory comments here will generally ignore the batch dim
        #   X is C x D; each row is a token emb
        #   W_k, W_q, W_v are D x H
        #   K = XW_k, Q = XW_q, V = XW_v are all C x H
        #   Each row of K is the key for a particular token
        #   Each row of Q is the query for a particular token
        B, C, D = X.shape
        K = self.key(X)  # (B, C, H)
        Q = self.query(X)  # (B, C, H)
        V = self.value(X)  # (B, C, H)
        
        # (i, j) entry of att is how much ith token attends to jth token (pre-normalization)
        # ith row of att is how much ith token attends to others (pre-normalization)
        # jth col of att is how much jth token attended to by others
        # thus, (i, j) entry must come from ith query, jth key. Hence QK^T
        att = Q @ K.transpose(-2,-1) # (B, C, H) @ (B, H, C) -> (B, C, C)
        att /= K.shape[-1]**0.5  # normalize by sqrt(H)
        
        # mask so that tokens can't attend to future
        att = att.masked_fill(self.tril[:C, :C] == 0, float('-inf')) # (B, C, C)
        # normalization is done along each row (amount ith attends must sum to 1)
        att = F.softmax(att, dim=-1) # (B, C, C)
        # ith row of output is convex combination of rows of value matrix,
        # where weights of convex combination come from ith row of att
        out = att @ V # (B, C, C) @ (B, C, H) -> (B, C, H)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        D, C, num_heads = config.D, config.C, config.num_heads
        if D % num_heads != 0:
            raise ValueError(f"{D=} not divisible by {num_heads=}")
        H = D // num_heads
        self.heads = nn.ModuleList([AttentionHead(H, D, C) for _ in range(num_heads)])
        self.proj = nn.Linear(H * num_heads, D)

    def forward(self, X: torch.tensor) -> torch.tensor:
        # Each head of attention writes to a disjoint subspace first, then we mix
        # (B, C, D) -> {cat[(B, C, H) (D/H) times} -> (B, C, D)
        head_outputs = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.proj(head_outputs)
        return X + out  # residual update

class MLP(nn.Module):
    """
    Ordinary Dense MLP layer.
    Result is added to the residual stream after multiplication by config.lambda_coeff.
    (B, C, D) -> (B, C, D') -> (B, C, D)
    """
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.D, config.b),
            nn.ReLU(),
            nn.Linear(config.b, config.D)
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        # (B, C, D) -> (B, C, D)
        return X + self.config.lambda_coeff * self.net(X)

