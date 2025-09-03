import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm  # assumes available

def _coalesce_coo(indices, values, size):
    coo = torch.sparse_coo_tensor(indices, values, size=size).coalesce()
    return coo.indices(), coo.values()

def _topk_to_coo_pos(X, k):
    """
    ReLU THEN top-k by value.
    X: (B, T, H, Nlift)
    Returns COO (2, nnz), values (nnz,), size = (BHT, BH*Nlift).
    """
    X = F.relu(X)  # nonnegative => "by value" == "by magnitude"
    B, T, H, N = X.shape
    G, BHT = B * H, B * H * T
    Xf = X.permute(0, 2, 1, 3).reshape(BHT, N)   # (BHT, N)
    k = min(k, N)
    vals, col_local = torch.topk(Xf, k=k, dim=-1)  # (BHT, k)
    rows = torch.arange(BHT, device=X.device).unsqueeze(1).expand(-1, k).reshape(-1)
    groups = rows // T                              # 0..(B*H-1)
    cols = col_local.reshape(-1) + groups * N       # block-diagonal column offset
    idx = torch.stack([rows, cols], dim=0)          # (2, nnz)
    return idx, vals.reshape(-1), torch.Size([BHT, G * N])

class SparseLiftAttention(nn.Module):
    """
    Lifted top-t sparse attention using spspmm (COO) for QK^T.
    - Q,K: project -> ReLU -> top-k by value -> COO
    - A = Q @ K^T via spspmm (autograd to Q/K values)
    - mask (0/1) + causal by filtering COO entries
    - p=1 normalization + learned sink (beta)
    - final CSR @ dense V
    """
    def __init__(self, d_model, num_heads, lift, t, causal=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.hd = d_model // num_heads
        self.lift = lift
        self.t = t
        self.causal = causal

        Nlift = lift * self.hd
        self.W_q = nn.Linear(d_model, num_heads * Nlift, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * Nlift, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * self.hd,  bias=False)
        self.W_o = nn.Linear(num_heads * self.hd, d_model,  bias=False)

        self.sink = nn.Parameter(torch.zeros(num_heads, self.hd))
        self.log_beta = nn.Parameter(torch.tensor(0.0))  # beta = exp(log_beta) >= 0

    def forward(self, x):
        """
        x: (B, T, D). Always applies causal masking internally (keep j <= i).
        """
        B, T, D = x.shape
        H, Hd, L = self.h, self.hd, self.lift
        Nlift = L * Hd
        G, BHT, BHNlift = B * H, B * H * T, B * H * Nlift
        device, dtype = x.device, x.dtype

        # Projections
        q = self.W_q(x).view(B, T, H, Nlift)
        k = self.W_k(x).view(B, T, H, Nlift)
        v = self.W_v(x).view(B, T, H, Hd)  # (B,T,H,Hd)

        # ReLU → top-k by value → COO over block-diagonal lifted cols
        q_idx, q_val, q_size = _topk_to_coo_pos(q, self.t)   # (2, nq), (nq,), size=(BHT, G*Nlift)
        k_idx, k_val, k_size = _topk_to_coo_pos(k, self.t)

        # Sparse@sparse → sparse (COO): A = Q @ K^T


        # A = Q @ K^T via spspmm  (all COO)
        # Q: (BHT x BHNlift)   K^T: (BHNlift x BHT)
        A_idx, A_val = spspmm(
            q_idx, q_val,          # Q (COO)
            k_idx.flip(0), k_val,  # K^T (COO) -> flip rows/cols
            BHT, BHNlift, BHT
        )
        A_size = torch.Size([BHT, BHT])

        # --- causal mask (keep j <= i) by filtering COO entries ---
        rows, cols = A_idx[0], A_idx[1]        # (nnz,)
        i = rows % T
        j = cols % T
        keep = (j <= i)
        rows, cols, A_val = rows[keep], cols[keep], A_val[keep]
        A_idx = torch.stack([rows, cols], dim=0)

        # Coalesce duplicates
        A_idx, A_val = _coalesce_coo(A_idx, A_val, A_size)

        # Build CSR row_ptr (sorted by row)
        order = torch.argsort(A_idx[0])
        rows, cols, A_val = A_idx[0][order], A_idx[1][order], A_val[order]
        counts = torch.bincount(rows, minlength=BHT)
        row_ptr = torch.zeros(BHT + 1, device=device, dtype=torch.int64)
        row_ptr[1:] = torch.cumsum(counts, 0)

        # Row-wise denom + sink
        denom = torch.zeros(BHT, device=device, dtype=dtype)
        denom.index_add_(0, rows, A_val)
        beta = self.log_beta.exp()
        denom_ws = denom + beta

        # Normalize in COO
        A_val = A_val / denom_ws[rows].clamp_min(1e-12)

        # CSR for fast SpMM with V
        A_csr = torch.sparse_csr_tensor(row_ptr, cols, A_val, size=A_size, dtype=dtype, device=device)

        # Aggregate values
        V_flat = v.permute(0, 2, 1, 3).reshape(BHT, Hd)  # (BHT, Hd)
        Y_flat = torch.sparse.mm(A_csr, V_flat)

        # Sink (per-head)
        head_idx = (torch.arange(BHT, device=device) // T) % H
        Y_flat = Y_flat + (beta / denom_ws).unsqueeze(-1) * self.sink[head_idx]

        # Back to (B, T, D)
        Y = Y_flat.view(B, H, T, Hd).permute(0, 2, 1, 3).reshape(B, T, H * Hd)
        return self.W_o(Y)

print("Imported")  # TODO - test only
