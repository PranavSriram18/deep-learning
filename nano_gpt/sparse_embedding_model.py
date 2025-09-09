# nano_gpt/models/sparse_embedding_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple

# --------- Identity-backward STE for sparse-forward projection ---------
class _TopKSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mask: torch.Tensor):
        # Sparse forward: keep only masked coords
        # No need to save tensors for backward; we want dense grad to x
        return x * mask

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # Dense backward: pass grad straight through to x; no grad for mask
        return grad_out, None

def ste_project_identity_backward(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _TopKSTE.apply(x, mask)


@dataclass
class SparseEmbeddingModelConfig:
    # model hyperparameters
    vocab_size: int = 40000
    dim: int = 128
    t_target: int = 16
    t_start: Optional[int] = 128
    total_anneal_steps: Optional[int] = 1024
    temperature: float = 1.0
    weight_std: float = 0.01

    # training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    print_every: int = 256
    train_steps: int = 8000
    context_length: int = 64
    sample_prompts: set[str] = field(default_factory=lambda: {"India is", "The United States is"})
    sample_length: int = 200

# --------- Model ---------
class SparseEmbeddingModel(nn.Module):
    """
    Minimal bigram LM with learned t-sparse embeddings (both sides), untied tables.

    - Input:  Z[v] -> TopK_t(Z[v]) used as embedding for token v
    - Output: U[v] -> TopK_t(U[v]) used as unembedding for token v
    - Forward computes logits = E @ Uproj^T and CE loss vs next tokens.
    - TopK uses identity-backward STE (dense grads to Z, U).
    - t is annealed linearly from t_start -> t_target over total_anneal_steps,
      but only when model.training = True.

    Shapes:
      xb, yb: (B, T) long
      logits: (B, T, V)
    """
    def __init__(
        self,
        config: SparseEmbeddingModelConfig,
    ):
        super().__init__()
        assert 1 <= config.t_target <= config.dim
        self.V = int(config.vocab_size)
        self.D = int(config.dim)

        self.t_target = int(config.t_target)
        self.t_start  = int(config.t_start) if config.t_start is not None else self.D
        self.t        = self.t_start
        self.total_anneal_steps = int(config.total_anneal_steps) if config.total_anneal_steps is not None else None
        self.temperature = float(config.temperature)

        # Parameters (untied)
        self.Z = nn.Parameter(torch.randn(self.V, self.D) * config.weight_std)  # input table
        self.U = nn.Parameter(torch.randn(self.V, self.D) * config.weight_std)  # output table

        # Step counter for annealing (CPU/long buffer)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    # --- utilities ---
    @staticmethod
    def _row_topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
        """
        x: (N, D) -> bool mask (N, D) of per-row top-k by |x|.
        Handles k >= D by returning all-True.
        """
        N, D = x.shape
        if k >= D:
            return torch.ones_like(x, dtype=torch.bool)
        # topk indices by abs value
        idx = torch.topk(x.abs(), k, dim=-1, sorted=False).indices  # (N, k)
        m = torch.zeros_like(x, dtype=torch.bool)
        m.scatter_(1, idx, True)
        return m

    def _maybe_anneal_t(self) -> None:
        """
        Linear t schedule from t_start to t_target over total_anneal_steps.
        Only advances when self.training is True.
        """
        if not self.training:
            return
        if self.total_anneal_steps is None or self.total_anneal_steps <= 0:
            self.t = self.t_target
            self._step += 1
            return
        s = int(self._step.item())
        p = min(1.0, s / self.total_anneal_steps)
        t_now = int(round(self.t_start + p * (self.t_target - self.t_start)))
        self.t = max(1, min(self.D, t_now))
        self._step += 1

    # --- main forward ---
    def forward(self, xb: torch.Tensor, yb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xb, yb: (B, T) longs
        Returns logits (B, T, V) and CE loss.
        """
        # Update t if training
        self._maybe_anneal_t()

        B, T = xb.shape
        V, D, t = self.V, self.D, self.t

        # Input side: gather Z rows and sparsify with TopK (sparse forward, dense back)
        x_flat = xb.reshape(-1)                 # (B*T,)
        z_rows = self.Z[x_flat]                 # (B*T, D)
        in_mask = self._row_topk_mask(z_rows, t)
        e_rows  = ste_project_identity_backward(z_rows, in_mask)  # (B*T, D)
        E = e_rows.view(B, T, D)                # (B, T, D)

        # Output side: sparsify all rows of U (per-row TopK)
        out_mask = self._row_topk_mask(self.U, t)     # (V, D) bool
        Uproj    = ste_project_identity_backward(self.U, out_mask)  # (V, D)

        # Logits
        logits = torch.matmul(E, Uproj.T) / max(1e-6, self.temperature)  # (B, T, V)

        # Loss
        loss = F.cross_entropy(logits.view(-1, V), yb.view(-1))
        return logits, loss

    # --- generation (bigram roll) ---
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,        # (B, T) context; we use last token
        max_new_tokens: int,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Bigram sampler p(next | current). Uses current t (no anneal in eval).
        """
        device = next(self.parameters()).device
        temperature = float(temperature or self.temperature)

        for _ in range(max_new_tokens):
            x_cur = idx[:, -1]                              # (B,)
            z = self.Z[x_cur]                               # (B, D)
            in_mask = self._row_topk_mask(z, self.t)
            e = ste_project_identity_backward(z, in_mask)   # (B, D)

            out_mask = self._row_topk_mask(self.U, self.t)  # (V, D)
            Uproj    = ste_project_identity_backward(self.U, out_mask)  # (V, D)

            logits = (e @ Uproj.T) / max(1e-6, temperature) # (B, V)

            if top_k is not None:
                topv, topi = torch.topk(logits, k=top_k, dim=-1)
                filt = torch.full_like(logits, float("-inf"))
                filt.scatter_(1, topi, topv)
                logits = filt

            probs = F.softmax(logits, dim=-1)               # (B, V)
            next_token = torch.multinomial(probs, 1)        # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
