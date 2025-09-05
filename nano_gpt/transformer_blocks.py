
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import functional as F # type: ignore

class Head(nn.Module):
    """
    One head of self-attention.
    H: head size 
    D: embedding dimension of X
    C: context length (block size)
    dropout: dropout parameter
    """
    def __init__(self, H: int, D: int, C: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(D, H, bias=False)
        self.query = nn.Linear(D, H, bias=False)
        self.value = nn.Linear(D, H, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(C, C)))
        self.dropout = nn.Dropout(dropout)

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
        att = self.dropout(att)
        # ith row of output is convex combination of rows of value matrix,
        # where weights of convex combination come from ith row of att
        out = att @ V # (B, C, C) @ (B, C, H) -> (B, C, H)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads: int, D: int, C: int, dropout: float = 0.):
        super().__init__()
        H = D // num_heads
        self.heads = nn.ModuleList([Head(H, D, C, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(H * num_heads, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.tensor) -> torch.tensor:
        # Each head of attention writes to a disjoint subspace first, then we mix
        # (B, C, D) -> {cat[(B, C, H) (D/H) times} -> (B, C, D)
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class MLP(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    (B, C, D) -> (B, C, D)
    """
    def __init__(self, D: int, ff_expansion: int = 4, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, ff_expansion * D),
            nn.ReLU(),
            nn.Linear(ff_expansion * D, D),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        # (B, C, D) -> (B, C, D)
        return self.net(X)

class Block(nn.Module):
    """
    Transformer block: communication followed by computation.
    (B, C, D) -> (B, C, D)
    """

    def __init__(self, D: int, num_heads: int, C: int, ff_expansion: int = 4, dropout: float = 0.):
        """
        D: embedding dimension
        num_heads: the number of heads we'd like
        C: context length
        ff_expansion: ratio of hidden dim to input dim of feedforward block
        dropout: dropout parameter
        """
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, D, C, dropout)
        self.ffwd = MLP(D=D, ff_expansion=ff_expansion, dropout=dropout)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

    def forward(self, X: torch.tensor) -> torch.tensor:
        # (B, C, D) -> (B, C, D)
        # {norm, op, residual} for op in {attn, ff}
        X = X + self.attn(self.ln1(X))
        X = X + self.ffwd(self.ln2(X))
        return X
        