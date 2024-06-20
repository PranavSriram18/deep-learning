
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import functional as F

class Head(nn.Module):
    """ One head of self-attention """

    """
    H: head size 
    D: embedding dimension of X
    C: context length (block size)
    dropout: dropout parameter
    """
    def __init__(self, H, D, C, dropout):
        super().__init__()
        self.key = nn.Linear(D, H, bias=False)
        self.query = nn.Linear(D, H, bias=False)
        self.value = nn.Linear(D, H, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(C, C)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # Linear projections here are all (B, T, D) -> (B, T, H)
        # Explanatory comments here will generally ignore the batch dim
        #   X is T x D; each row is a token emb
        #   W_k, W_q, W_v are D x H
        #   K = XW_k, Q = XW_q, V = XW_v are all T x H
        B,T,D = X.shape
        k = self.key(X)  # (B, T, H)
        q = self.query(X)  # (B, T, H)
        v = self.value(X)  # (B, T, H)
        # compute attention scores
        # Ignoring batch dim, queries, keys are stored in rows. So attention
        # is calculated as QK^T
        att = q @ k.transpose(-2,-1) # (B, T, H) @ (B, H, T) -> (B, T, T)
        att = att * k.shape[-1]**-0.5  # normalize by sqrt of head size

        # ignore batch dim for a moment, and consider last two dims of att
        # ith row of att is how much ith token attends to others (pre-normalization)
        # (i, j) entry is how much ith token attends to jth token (pre-normalization)
        # mask so that tokens can't attend to future
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # normalize along each row
        att = F.softmax(att, dim=-1) # (B, T, T)
        att = self.dropout(att)
        # ith row of output is convex combination of rows of value matrix,
        # where weights of convex combination come from ith row of att
        out = att @ v # (B, T, T) @ (B, T, H) -> (B, T, H)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, H, D, C, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(H, D, C, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(H * num_heads, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, D, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.ReLU(),
            nn.Linear(4 * D, D),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, D, n_head, C, dropout):
        """
        D: embedding dimension
        n_head: the number of heads we'd like
        C: context length
        dropout: dropout parameter
        """
        super().__init__()
        H = D // n_head
        self.attn = MultiHeadAttention(n_head, H, D, C, dropout)
        self.ffwd = FeedFoward(D, dropout)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

    def forward(self, x):
        # norm, attn, residual, norm, feedforward, residual
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        