import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn import functional as F

from nano_gpt.transformer_blocks import MultiHeadAttention

class ChunkedAttention(nn.Module):
    """
    Key tensor dimensions:
    B: batch size
    C: full context length
    M: block size
    k: Number of blocks
    H: head size 
    D: embedding dimension of X
    """

    def __init__(self, C: int, M: int, D: int, num_heads: int, dropout: float = 0.):
        super().__init__()
        self.C = C
        self.M = M
        self.D = D
        assert self.C % self.M == 0, "self.C must be divisible by self.M."
        self.k = self.C // self.M
        self.num_heads = num_heads 
        self.dropout = nn.Dropout(dropout)
        # context length for MHA is chunk size, not full context length
        self.mha = MultiHeadAttention(num_heads, D, M)  
        self.proj = nn.Linear(self.M, 1, bias=False)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        BxCxD -> BxkxD
        """
        B, C, D = X.shape
        if C < self.C:
            X = F.pad(X, (0, 0, self.C-C, 0))  # B x self.C x D
        elif C > self.C:
            X = X[:, 0:self.C, :]  # B x self.C x D
        C = self.C
        # TODO - add assertion on D (though not having it is also okay as it
        # adds flexibility)
        # split context into k chunks each of length M
        X = X.view(B, self.k, self.M, D)  # B x k x M x D
        X_rs = X.reshape(B * self.k, self.M, D)  # (B*k)xMxD
        # apply attention to each chunk independently in parallel
        block_outputs = self.mha(X_rs)  # (B*k)xMxD
        # Each block will have a single embedding of dimension D
        block_outputs_rs = block_outputs.permute(0, 2, 1)  # (B*k)xDxM
        block_outputs_reduced = self.proj(block_outputs_rs).squeeze(2)  # (B*k)xD
        out = block_outputs_reduced.reshape(B, self.k, D)  # BxkxD
        out = self.dropout(out)  # BxkxD
        return out
