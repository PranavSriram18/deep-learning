
from layers.layer_utils import AUX_LOSS_SUFFIX
import torch  # type: ignore
from torch import nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.nn.utils.parametrize as parametrize  # type: ignore

from layers.layer_config import MLPConfig
from layers.layer_utils import UnitColNorm


class SparseRead(nn.Module):
    """
    Read module: x in R^D -> h = V^T x in R^(m*b), grouped as (m, b).

    V is parameterized to have unit-norm columns over the D dimension.
    """

    def __init__(self, D: int, m: int, b: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.D = D
        self.m = m
        self.b = b
        self.dtype = dtype

        # V: (D, m, b)
        self.V = nn.Parameter(torch.randn(self.D, self.m, self.b, dtype=self.dtype))
        parametrize.register_parametrization(self, "V", UnitColNorm(dim=0))

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: (N, D)
        returns h_all: (N, m, b)
        """
        return torch.einsum("nd,dmb->nmb", x_flat, self.V)


class TopKAutoencodeInhibitor(nn.Module):
    """
    Lateral inhibition via top-k expert selection by energy, with an autoencoding
    auxiliary loss using the read dictionary V for reconstruction.

    Selection score per expert i: E_i^2 = ||h_i||^2, where h_i = V_i^T x.
    Reconstruction: x_hat = sum_{i in S} V_i h_i.
    Aux loss: ||x - x_hat||^2 (mean over batch).
    """

    def __init__(self, D: int, m: int, b: int, k: int, eps: float = 1e-8):
        super().__init__()
        self.D = D
        self.m = m
        self.b = b
        self.k = k
        self.eps = eps

        if not (1 <= self.k <= self.m):
            raise ValueError(f"k must satisfy 1 <= k <= m, got k={k}, m={m}")

    @torch.no_grad()
    def _gather_V_active(self, V: torch.Tensor, topk_idxs: torch.Tensor) -> torch.Tensor:
        """
        V: (D, m, b)
        topk_idxs: (N, k)
        returns V_active: (N, k, D, b)
        """
        N = topk_idxs.size(0)
        V_t = V.permute(1, 0, 2)  # (m, D, b)
        idx_exp = topk_idxs.view(N, self.k, 1, 1).expand(-1, -1, self.D, self.b)
        return torch.gather(V_t.unsqueeze(0).expand(N, -1, -1, -1), 1, idx_exp)

    def forward(
        self,
        x_flat: torch.Tensor,
        h_all: torch.Tensor,
        V: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        x_flat: (N, D) assumed pre-normalized
        h_all: (N, m, b)
        V: (D, m, b)

        returns:
          h_sparse: (N, k, b)
          topk_idxs: (N, k)
          aux: dict of scalars/tensors
        """
        # Energy per expert: (N, m)
        energy = (h_all * h_all).sum(dim=-1)

        # Top-k experts by energy
        topk_vals, topk_idxs = energy.topk(self.k, dim=-1)  # (N, k)

        # Gather coefficients for selected experts: (N, k, b)
        idxs_expand = topk_idxs.unsqueeze(-1).expand(-1, -1, self.b)
        h_sparse = torch.gather(h_all, 1, idxs_expand)

        # Reconstruction using V as decoder: x_hat = sum_i V_i h_i
        V_active = self._gather_V_active(V, topk_idxs)  # (N, k, D, b)
        x_hat = torch.einsum("nkdb,nkb->nd", V_active, h_sparse)  # (N, D)

        # Uncaptured energy: ||x - x_hat||^2 per token
        # since x is unit norm, scale is O(1)
        resid = x_flat - x_hat
        uncaptured_energy = (resid * resid).sum(dim=-1).mean()

        # Load balancing metric: entropy of average per-expert energy distribution
        # higher is better; max possible is 1
        avg_energy_per_expert = energy.mean(dim=0)  # (m,)
        denom = avg_energy_per_expert.sum().clamp_min(self.eps)
        probs = (avg_energy_per_expert / denom).clamp_min(self.eps)  # (m,)
        balance_entropy = -(probs * torch.log(probs)).sum() / torch.log(torch.tensor(self.m, dtype=probs.dtype, device=probs.device))

        # Captured energy proxies
        captured_energy_proj = topk_vals.sum(dim=-1).mean()
        recon_energy = (x_hat * x_hat).sum(dim=-1).mean()

        aux = {
            "captured_energy_proj": captured_energy_proj,
            "recon_energy": recon_energy,
            "uncaptured_energy": uncaptured_energy,
            "balance_entropy": balance_entropy,
            "topk_aux_loss": uncaptured_energy + 0.5 * (1. - balance_entropy),
        }
        return h_sparse, topk_idxs, aux


class DenseWrite(nn.Module):
    """
    Write module: sparse (k, b) activations -> dense R^D using expert-specific U_i.

    U is left unconstrained initially.
    """

    def __init__(self, D: int, m: int, b: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.D = D
        self.m = m
        self.b = b
        self.dtype = dtype

        # U: (m, D, b) with unit norm along D for each writer vector
        self.U = nn.Parameter(torch.randn(self.m, self.D, self.b, dtype=self.dtype))
        parametrize.register_parametrization(self, "U", UnitColNorm(dim=1))

    def forward(self, h_sparse: torch.Tensor, topk_idxs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        h_sparse: (N, k, b)
        topk_idxs: (N, k)
        returns:
          writes: (N, D)
          aux: dict including writer_aux_loss: scalar
        """
        N, k, b = h_sparse.shape
        if b != self.b:
            raise ValueError(f"Expected b={self.b}, got b={b}")

        idx_U = topk_idxs.view(N, k, 1, 1).expand(-1, -1, self.D, self.b)  # (N, k, D, b)
        U_active = torch.gather(self.U.unsqueeze(0).expand(N, -1, -1, -1), 1, idx_U)
        # Forward write: U_S h  -> (N, D)
        writes = torch.einsum("nkdb,nkb->nd", U_active, h_sparse)

        # Autoencode in h-space with stopgrad on h:
        # h_hat = U_S^T (U_S h), compare to h (detached)
        # U_S^T writes: (N, k, b)
        h_recon = torch.einsum("nkdb,nd->nkb", U_active, writes)
        h_target = h_sparse.detach()
        writer_recon_loss = (h_recon - h_target).pow(2).mean()

        aux = {"writer_aux_loss": writer_recon_loss}
        return writes, aux


class SparseExpertV3(nn.Module):
    """
    Wires together Read, Lateral Inhibition, and Write.

    Forward:
      x_in -> normalize -> h_all = V^T x
      (h_sparse, idxs, aux) = inhibitor(x, h_all, V)
      writes = U_S h_sparse
      x_out = normalize(x + alpha * writes)
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.D = config.D
        self.b = config.b
        self.m = config.m
        self.k = config.k

        self.dtype = torch.float32

        self.alpha = getattr(config, "write_scale", 1.0)
        self.eps = getattr(config, "norm_eps", 1e-8)

        self.reader = SparseRead(self.D, self.m, self.b, dtype=self.dtype)
        self.inhibitor = TopKAutoencodeInhibitor(self.D, self.m, self.b, self.k, eps=self.eps)
        self.writer = DenseWrite(self.D, self.m, self.b, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (..., D)
        returns:
          x_out: (..., D) with unit norm per vector
          aux: dict of auxiliary metrics/loss
        """
        input_shape = x.shape
        if input_shape[-1] != self.D:
            raise ValueError(f"Expected last dim D={self.D}, got {input_shape[-1]}")

        x_flat = x.reshape(-1, self.D)
        x_flat = F.normalize(x_flat, p=2, dim=-1, eps=self.eps)

        # 1) Read
        h_all = self.reader(x_flat)  # (N, m, b)

        # 2) Lateral inhibition + aux loss (autoencode with V)
        h_sparse, topk_idxs, topk_aux = self.inhibitor(x_flat, h_all, self.reader.V)

        # 3) Write
        writes, writer_aux = self.writer(h_sparse, topk_idxs)  # (N, D), aux

        # Residual update + normalize
        x_out = x_flat + (self.alpha * writes)
        x_out = F.normalize(x_out, p=2, dim=-1, eps=self.eps)

        x_out = x_out.view(*input_shape)
        aux: dict = topk_aux | writer_aux
        aux[AUX_LOSS_SUFFIX] = aux["topk_aux_loss"] + aux["writer_aux_loss"]
        return x_out, aux
