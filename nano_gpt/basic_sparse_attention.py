from nano_gpt.transformer_blocks import MLP
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from torch.nn.utils import parametrize  # type: ignore

class UnitNorm(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # Normalize along `dim` so each row vector (output neuron) has L2 norm 1
        return W / W.norm(dim=self.dim, keepdim=True).clamp_min(self.eps)

class BasicSparseAttentionHead(nn.Module):
    """
    Single-head causal self-attention with lifted sparse Q/K.

    H: head size in the value/output space (same as vanilla head size)
    D: embedding dimension of inputs X
    C: context length (block size)
    alpha: lift factor; lifted dimension D2 = alpha * H
    t: number of active coordinates kept per token in the lifted space

    Implementation:
      - Q_lift = X @ W_q_lift in R^{D2}, K_lift = X @ W_k_lift in R^{D2}
      - Per token, keep top-|t| entries by absolute value in Q_lift and K_lift; others set to 0
      - Attention scores use dot products in lifted space
      - Scale by sqrt(t) (effective active dimensionality)
      - Values remain in the original head size H
    """
    def __init__(self, H: int, D: int, C: int, alpha: float = 4.0, t: int = 32):
        super().__init__()
        self.H = H
        self.D = D
        self.C = C
        self.alpha = alpha
        self.D2 = int(alpha * H)
        self.t = max(1, min(t, self.D2))

        self.key_lift   = nn.Linear(D, self.D2, bias=False)
        self.query_lift = nn.Linear(D, self.D2, bias=False)
        self.value      = nn.Linear(D, H, bias=False)

        # Enforce unit L2 norm on each output neuron’s weight vector (rows)
        parametrize.register_parametrization(self.key_lift,   "weight", UnitNorm(dim=1))
        parametrize.register_parametrization(self.query_lift, "weight", UnitNorm(dim=1))

        self.register_buffer("tril", torch.tril(torch.ones(C, C)))

    @staticmethod
    def _top_t(x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Keep top-t entries by absolute value along the last dimension.
        Remaining entries are set to zero. Gradients flow through selected entries.
        """
        if t <= 0:
            return torch.zeros_like(x)
        last = x.size(-1)
        if t >= last:
            return x
        idx = x.abs().topk(k=t, dim=-1).indices
        mask = torch.zeros_like(x, dtype=torch.bool).scatter_(-1, idx, True)
        return torch.where(mask, x, torch.zeros_like(x))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, C, D)
        B, C, _ = X.shape

        V      = self.value(X)       # (B, C, H)

        # lift then sparsify queries, keys
        K_lift = self.key_lift(X)    # (B, C, D2)
        Q_lift = self.query_lift(X)  # (B, C, D2)
        K = self._top_t(K_lift, self.t)  # (B, C, D2)
        Q = self._top_t(Q_lift, self.t)  # (B, C, D2)

        # from here basically follow vanilla attention (but normalize by effective active dimensionality)
        att = Q @ K.transpose(-2, -1)   # (B, C, C)
        att = att / (self.t ** 0.5)
        att = att.masked_fill(self.tril[:C, :C] == 0, float("-inf"))  # (B, C, C)
        att = F.softmax(att, dim=-1)                                   # (B, C, C)

        out = att @ V  # (B, C, H)
        return out

class BasicSparseAttention(nn.Module):
    """ multiple heads of basic sparse self-attention in parallel """

    def __init__(self, num_heads: int, D: int, C: int, alpha: float, t: int):
        super().__init__()
        assert D % num_heads == 0
        H = D // num_heads
        self.heads = nn.ModuleList([BasicSparseAttentionHead(H, D, C, alpha, t) for _ in range(num_heads)])
        self.proj = nn.Linear(D, D)

    def forward(self, X: torch.tensor) -> torch.tensor:
        # Each head of attention writes to a disjoint subspace of output
        # (B, C, D) -> {cat[(B, C, H) (D/H) times} -> (B, C, D)
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.proj(out)  # (B, C, D)
        return out

class BasicSparseTransformerBlock(nn.Module):
    """
    Transformer block: communication followed by computation.
    (B, C, D) -> (B, C, D)
    """

    def __init__(self, D: int, num_heads: int, C: int, alpha: float, t: int, ff_expansion: int = 4, dropout: float = 0.):
        """
        D: embedding dimension
        num_heads: the number of heads we'd like
        C: context length
        alpha: lift factor; lifted dimension D2 = alpha * H (H is head dim)
        t: number of active coordinates kept per token in the lifted space
        ff_expansion: ratio of hidden dim to input dim of feedforward block
        dropout: dropout parameter (for MLP)
        """
        super().__init__()
        self.attn = BasicSparseAttention(num_heads, D, C, alpha, t)
        self.ffwd = MLP(D, ff_expansion, dropout)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

    def forward(self, X: torch.tensor) -> torch.tensor:
        # (B, C, D) -> (B, C, D)
        # {norm, op, residual} for op in {attn, ff}
        X = X + self.attn(self.ln1(X))
        X = X + self.ffwd(self.ln2(X))
        return X
        