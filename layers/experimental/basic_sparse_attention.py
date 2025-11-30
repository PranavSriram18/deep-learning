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
    Hard top-t in forward; optional straight-through estimator (STE) in backward.
    """
    def __init__(
        self,
        H: int, D: int, C: int,
        alpha: float = 4.0,
        t: int = 32,
        use_ste: bool = False,
    ):
        super().__init__()
        self.H, self.D, self.C = H, D, C
        self.alpha = alpha
        self.D2 = int(alpha * H)
        self.t = max(1, min(t, self.D2))
        self.use_ste = use_ste

        self.key_lift   = nn.Linear(D, self.D2, bias=False)
        self.query_lift = nn.Linear(D, self.D2, bias=False)
        self.value      = nn.Linear(D, H,   bias=False)

        # Unit-norm rows for lifted projections (direction vectors)
        parametrize.register_parametrization(self.key_lift,   "weight", UnitNorm(dim=1))
        parametrize.register_parametrization(self.query_lift, "weight", UnitNorm(dim=1))

        self.register_buffer("tril", torch.tril(torch.ones(C, C)))

    @staticmethod
    def _top_t(x: torch.Tensor, t: int) -> torch.Tensor:
        if t <= 0:
            return torch.zeros_like(x)
        last = x.size(-1)
        if t >= last:
            return x
        idx = x.abs().topk(k=t, dim=-1).indices
        mask = torch.zeros_like(x, dtype=torch.bool).scatter_(-1, idx, True)
        return torch.where(mask, x, torch.zeros_like(x))

    def _sparsify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hard top-t in forward. If use_ste, gradients flow as identity:
        z = x + (y - x).detach() makes forward equal y, backward dL/dx = dL/dy.
        """
        y = self._top_t(x, self.t)
        if self.use_ste:
            return x + (y - x).detach()
        return y

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, C, D)
        B, C, _ = X.shape

        V      = self.value(X)        # (B, C, H)
        Q_lift = self.query_lift(X)   # (B, C, D2)
        K_lift = self.key_lift(X)     # (B, C, D2)

        # Hard-sparse tensors in forward; STE controls backward
        Q = self._sparsify(Q_lift)    # (B, C, D2)
        K = self._sparsify(K_lift)    # (B, C, D2)

        # Attention over lifted space
        att = Q @ K.transpose(-2, -1)     # (B, C, C)
        att = att / (self.t ** 0.5)
        att = att.masked_fill(self.tril[:C, :C] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ V                     # (B, C, H)
        return out

class BasicSparseAttention(nn.Module):
    """ Multiple heads of basic sparse self-attention in parallel. """
    def __init__(self, D: int, num_heads: int, C: int, alpha: float, t: int, use_ste: bool = True):
        super().__init__()
        assert D % num_heads == 0
        H = D // num_heads
        self.heads = nn.ModuleList([
            BasicSparseAttentionHead(H=H, D=D, C=C, alpha=alpha, t=t, use_ste=use_ste)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(D, D, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(X) for h in self.heads], dim=-1)  # (B, C, D)
        out = self.proj(out)
        return out


class BasicSparseTransformerBlock(nn.Module):
    """
    Transformer block: communication followed by computation.
    (B, C, D) -> (B, C, D)
    """

    def __init__(
        self, 
        D: int, 
        num_heads: int, 
        C: int, 
        alpha: float, 
        t: int, 
        ff_expansion: int = 4, 
        dropout: float = 0., 
        use_ste: bool = True
    ):
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
        self.attn = BasicSparseAttention(D=D, num_heads=num_heads, C=C, alpha=alpha, t=t, use_ste=use_ste)
        self.ffwd = MLP(D=D, ff_expansion=ff_expansion, dropout=dropout)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

    def forward(self, X: torch.tensor) -> torch.tensor:
        # (B, C, D) -> (B, C, D)
        # {norm, op, residual} for op in {attn, ff}
        X = X + self.attn(self.ln1(X))
        X = X + self.ffwd(self.ln2(X))
        return X
        