import torch  # type: ignore
from torch import nn  # type: ignore
import torch.nn.utils.parametrize as parametrize  # type: ignore
from layers.layer_config import MLPConfig
from layers.layer_utils import UnitColNorm

class SparseExpertV2(nn.Module):
    """
    Sparse Expert layer with corrected-energy based reranking.
    """
    def __init__(
        self,
        config: MLPConfig
    ):
        super().__init__()
        self.D = config.D
        self.b = config.b
        self.m = config.m
        self.k = config.k
        # Draft pool size (e.g., 2*k). 
        # Ensure draft size is defined in config, or default to 2*k
        self.k_draft = getattr(config, 'k_draft', 2 * config.k)
        
        self.coherence_coeff = config.coherence_coeff
        self.lambda_coeff = config.lambda_coeff
        self.dtype = torch.float32

        # params
        # V: linear map D -> (m, b). Normalized columns.
        self.V = nn.Parameter(torch.randn(self.D, self.m, self.b)) 
        parametrize.register_parametrization(self, "V", UnitColNorm(dim=0))

        # U: linear map b -> D. Normalized columns.
        self.U = nn.Parameter(torch.randn(self.m, self.D, self.b)) 
        parametrize.register_parametrization(self, "U", UnitColNorm(dim=1))

        self.u_scales = nn.Parameter(torch.zeros(self.m, self.b)) 
    
    def U_eff(self):
        """
        Write matrix U scaled by scalar terms.
        """
        s = torch.tanh(self.u_scales) * self.lambda_coeff 
        return self.U * s.unsqueeze(1) 

    def forward(self, x: torch.Tensor):
        """
        x is (..., D). Let N denote product of trailing dims.
        """
        # Flatten batch dimensions for easier indexing during competition
        input_shape = x.shape
        x_flat = x.reshape(-1, self.D) # (N, D)
        
        # 1. READ: Compute all raw projections
        # Note: We use the flattened batch for simpler gather logic
        expert_reprs = torch.einsum("nd,dmb->nmb", x_flat, self.V)  # (N, m, b)

        # 2. DRAFT: Select Top-2k based on Raw Energy
        # Raw (Signed) Energy: E_i = v_i^T x; E_i^2 = ||v_i^Tx||^2
        raw_energy_sq = (expert_reprs ** 2).sum(dim=-1)  # (N, m)
        
        # Draft selection (Top P)
        draft_vals, draft_idxs = raw_energy_sq.topk(self.k_draft, dim=-1) # (N, P)
        
        # 3. RE-RANK: Compute Lateral Inhibition
        # Gather the V vectors for the draft candidates
        # V shape: (D, m, b) -> Transpose to (m, D, b) for gathering
        V_t = self.V.permute(1, 0, 2) # (m, D, b)
        
        # Gather logic: expand indices to match dimensions
        # draft_idxs: (N, P) -> (N, P, D, b)
        idxs_expanded = draft_idxs.view(x_flat.size(0), self.k_draft, 1, 1).expand(-1, -1, self.D, self.b)
        
        # V_draft: The read vectors for the top 2k experts
        # Shape: (N, P, D, b)
        V_draft = torch.gather(V_t.unsqueeze(0).expand(x_flat.size(0), -1, -1, -1), 1, idxs_expanded)
        
        # Prepare for Gram Matrix: Flatten (D, b) into a single vector per expert
        V_draft_flat = V_draft.reshape(x_flat.size(0), self.k_draft, -1)  # (N, P, D*b)
        
        # Normalize for accurate cosine similarity (Gram Matrix)
        # Since V columns were unit norm, the norm of the flattened vector is sqrt(b). 
        # We re-normalize to make diagonal of Gram = 1.
        V_draft_norm = torch.nn.functional.normalize(V_draft_flat, p=2, dim=-1)
        
        # Compute Local Gram Matrix (Interaction)
        # G_local: (N, 2k, 2k)
        G_local = torch.bmm(V_draft_norm, V_draft_norm.transpose(1, 2))
        
        # Compute Inhibition Term: sum_{j!=i} E_i * E_j * G_ij
        # We approximate E_i as sqrt(raw_energy_sq) (magnitude of projection)
        E_draft = draft_vals.sqrt() # (N, 2k)
        
        # Mask diagonal to 0
        mask = 1 - torch.eye(self.k_draft, device=x.device).unsqueeze(0)
        G_off_diag = G_local * mask
        
        # Inhibition: (N, 2k)
        # (N, 1, 2k) @ (N, 2k, 2k) -> (N, 1, 2k) -> squeeze
        inhibition = (E_draft.unsqueeze(1) @ G_off_diag).squeeze(1) * E_draft
        
        # Corrected Score: Raw Energy - Inhibition
        # Note: We use the squared energy for the score to match dimensions
        corrected_scores = draft_vals - inhibition

        # 4. FINAL SELECTION: Top-k from the draft pool
        final_vals_local, final_idxs_local = corrected_scores.topk(self.k, dim=-1) # (N, k)
        
        # Map local indices (0..2k) back to global expert indices (0..m)
        topk_idxs = torch.gather(draft_idxs, 1, final_idxs_local) # (N, k)
        
        # 5. WRITE: Project back to output dimension
        # Gather the specific expert representations (coefficients) for the final k
        # We use the raw calculated representations from step 1
        # expert_reprs: (N, m, b)
        
        idxs_repr_expand = topk_idxs.unsqueeze(-1).expand(-1, -1, self.b) # (N, k, b)
        expert_reprs_topk = torch.gather(expert_reprs, 1, idxs_repr_expand) # (N, k, b)
        
        # Gather Write Matrices U
        # U_eff: (m, D, b) -> (N, k, D, b) via gathering
        U_eff_all = self.U_eff() # (m, D, b)
        idx_U_expand = topk_idxs.view(x_flat.size(0), self.k, 1, 1).expand(-1, -1, self.D, self.b)
        U_active = torch.gather(U_eff_all.unsqueeze(0).expand(x_flat.size(0), -1, -1, -1), 1, idx_U_expand)

        # Compute output updates
        # (N, k, D, b) * (N, k, b) -> (N, D)
        # Sum over k (experts) and b (subspace)
        writes = torch.einsum('nkdb,nkb->nd', U_active, expert_reprs_topk)
        
        # Reshape back to original dimensions (..., D)
        writes = writes.view(*input_shape)

        # Calculate Aux terms
        # We pass the draft-stage info to calculate the accurate "Corrected Energy"
        aux_dict = self._compute_aux_terms(E_draft, G_local, final_idxs_local)
        
        return x + writes, aux_dict

    def _compute_aux_terms(
        self, 
        E_draft: torch.Tensor, 
        G_draft: torch.Tensor, 
        final_idxs_local: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Computes the corrected energy capture metrics.
        
        E_draft: (N, 2k) - Magnitude of projections for draft set
        G_draft: (N, 2k, 2k) - Gram matrix of draft set
        final_idxs_local: (N, k) - Indices of the winner experts within the draft set
        """
        N = E_draft.size(0)
        
        # Gather E and G for the final selected k experts
        # E_final: (N, k)
        E_final = torch.gather(E_draft, 1, final_idxs_local)
        
        # G_final: (N, k, k)
        # Need to gather rows and columns from the (2k, 2k) matrix
        # 1. Gather rows: (N, k, 2k)
        idx_rows = final_idxs_local.unsqueeze(2).expand(-1, -1, self.k_draft)
        G_rows = torch.gather(G_draft, 1, idx_rows)
        # 2. Gather cols: (N, k, k)
        idx_cols = final_idxs_local.unsqueeze(1).expand(-1, self.k, -1)
        G_final = torch.gather(G_rows, 2, idx_cols)
        
        # 1. Raw Energy (Sum of squares)
        raw_energy_batch = (E_final ** 2).sum()
        
        # 2. Overlap Energy (Off-diagonal interactions)
        mask = 1 - torch.eye(self.k, device=E_final.device).unsqueeze(0)
        G_off_diag = G_final * mask
        
        # E_i * G_ij * E_j
        overlap_batch = (E_final.unsqueeze(1) @ G_off_diag @ E_final.unsqueeze(2)).sum()
        
        # 3. Corrected Total Energy
        corrected_total_energy = (raw_energy_batch - overlap_batch) / N
        
        # 4. Aux Loss
        # Maximize energy captured (minimize negative energy)
        # Add coherence penalty (minimize overlap)
        # Note: Our formula implies maximizing (Raw - Overlap). 
        # So Loss = -(Raw - Overlap) = Overlap - Raw.
        aux_loss = -corrected_total_energy

        return {
            "total_energy": corrected_total_energy,
            "raw_energy": raw_energy_batch / N,
            "overlap_penalty": overlap_batch / N,
            "aux_loss": aux_loss
        }