import select
from layers.layer_config import MLPConfig
from layers.layer_utils import UnitColNorm
import torch  # type: ignore
from torch import nn  # type: ignore

import torch.nn.utils.parametrize as parametrize  # type: ignore

class SparseExpertLayer(nn.Module):
    def __init__(
        self,
        config: MLPConfig
    ):
        """
        Initializes a SparseExpertLayer.
        """
        super().__init__()
        self.D = config.D
        self.b = config.b
        self.m = config.m
        self.k = config.k
        self.k_f = config.k_f
        self.coherence_coeff = config.coherence_coeff
        self.lambda_coeff = config.lambda_coeff

        self.dtype = torch.float32

        # params
        self.V = nn.Parameter(torch.randn(self.D, self.m, self.b))  # linear map D -> (m, b)
        parametrize.register_parametrization(self, "V", UnitColNorm(dim=0))

        self.U = nn.Parameter(torch.randn(self.m, self.D, self.b))  # foreach expert, linear map b -> D
        parametrize.register_parametrization(self, "U", UnitColNorm(dim=1))

        self.u_scales = nn.Parameter(torch.zeros(self.m, self.b))  # pre-tanh scales
    
    def U_eff(self):
        s = torch.tanh(self.u_scales) * self.lambda_coeff  # (m, b) in [-lambda_coeff, lambda_coeff]
        return self.U * s.unsqueeze(1)  # broadcast on D

    def forward(self, x: torch.Tensor):
        """
        x is (..., D).
        Let N denote product of trailing dims (e.g. batch_size * num_tokens).
        """

        # TODO: consider if/where we want to pre-norm the input
        
        # read D -> (m, b) foreach (batch_elem, token)
        expert_reprs = torch.einsum("...d,dmb->...mb", x, self.V)  # (..., m, b)

        # score each of m experts
        repr_l2_sq = torch.einsum("...mb,...mb->...m", expert_reprs, expert_reprs)  # (..., m)
        topk_vals, topk_idxs = repr_l2_sq.topk(self.k, dim=-1)  # (..., k), (..., k)
        expanded_idxs = topk_idxs.unsqueeze(-1).expand(*topk_idxs.shape, self.b)  # (..., k, b)

        # (m expert reprs of size b -> k expert reprs of size b) foreach (batch_elem, token)
        expert_reprs_topk = torch.gather(expert_reprs, dim=-2, index=expanded_idxs)  # (..., m, b) -> (..., k, b)

        # b->D map foreach (batch_elem, token, active_expert)
        U_active = self.U_eff()[topk_idxs]  # (m, D, b) select (..., k) -> (..., k, D, b)

        # reduce over both b dimension (expert write) and k dimension (experts interact additively
        # within a (batch_elem, token) pair)
        writes = torch.einsum('...kdb,...kb->...d', U_active, expert_reprs_topk)  # (..., D)

        aux_dict: dict[str, torch.Tensor] = self._compute_aux_terms(topk_vals, topk_idxs)
        return x + writes, aux_dict

    def _compute_aux_terms(
        self, topk_vals: torch.Tensor, topk_idxs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with keys
        "energy", "v_coherence_penalty", "u_coherence_penalty", "layer_aux_loss"
        terms are scaled to be O(1) in reasonable-behavior case; upstream model can rescale as
        needed
        """
        # energy terms
        total_energy, energy_per_expert, relative_energy, select_rate = self._compute_captured_energy(
            topk_vals, topk_idxs)

        # coherence terms
        V_rs = self.V.reshape(self.D, self.m*self.b)  # (D, m*b)
        U_rs = self.U_eff().reshape(self.D, self.m*self.b)  # (D, m*b)
        I = torch.eye(self.m * self.b, device=V_rs.device, dtype=V_rs.dtype)

        def coherence_penalty(M):
            # very simple coherence penalty for now
            # M: shape (D, m*b)
            # average off-diag dot prod is ~sqrt(1/D) for random, ~1 for mode-collapse
            # scale by sqrt(D) so it's O(1) in randomly initialized state
            return torch.abs(M.T @ M - I).mean() * (self.D ** 0.5)

        v_coherence_penalty, u_coherence_penalty = coherence_penalty(V_rs), coherence_penalty(U_rs)

        # we'll experiment w more sophisticated versions later
        # for now, ignore u_coherence_penalty
        aux_loss = self.coherence_coeff * v_coherence_penalty - total_energy
        return {
            "total_energy": total_energy,
            "energy_per_expert": energy_per_expert,
            "relative_energy": relative_energy,
            "select_rate": select_rate,
            "v_coherence_penalty": v_coherence_penalty,
            "u_coherence_penalty": u_coherence_penalty,
            "aux_loss": aux_loss
        }

    def _compute_captured_energy(self, topk_vals, topk_idxs):
        # Shapes: topk_vals, topk_idxs: (..., k)
        m = self.m
        N = max(topk_vals.numel() // topk_vals.size(-1), 1)  # number of tokens N (batch * seq)
        dev = topk_vals.device
        dt  = topk_vals.dtype

        # flatten all leading dims
        vals_flat = topk_vals.reshape(-1)              # (N*k,)
        idxs_flat = topk_idxs.reshape(-1).to(torch.long)   # (N*k,)

        # total captured energy per expert (avg over all tokens that selected it)
        # note this is "raw" energy, without incoherence penalty, so could sum to > 1
        energy_per_expert = torch.zeros(m, device=dev, dtype=dt)
        energy_per_expert.scatter_add_(0, idxs_flat, vals_flat)   # (m,)
        energy_per_expert = energy_per_expert / N

        # how often each expert was selected
        select_rate = torch.zeros(m, device=dev, dtype=dt)
        select_rate.scatter_add_(0, idxs_flat, torch.ones_like(vals_flat))  # (m,)
        select_rate = select_rate / N

        # share of total captured energy
        total_energy = energy_per_expert.sum().clamp_min(1e-12)  # (1)
        relative_energy = energy_per_expert / total_energy  # (m,)

        return total_energy, energy_per_expert, relative_energy, select_rate


