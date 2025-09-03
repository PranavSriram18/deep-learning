import torch
import time

def timed_qk(C: int, D: int):
    """
    creates random matrices Q, K or size C x D
    times the calculation of QK^T
    prints the time taken
    """
    Q = torch.randn(C, D)
    K = torch.randn(C, D)
    start = time.time()
    ret = Q @ K.T
    print(f"Time taken for timed qk with {C=} {D=}: {time.time() - start} seconds")
    return ret

import time, torch

def _rand_k_sparse(c: int, n: int, k: int, *, device="cpu") -> torch.Tensor:
    cols = torch.rand(c, n, device=device).topk(k, dim=1).indices
    rows = torch.arange(c, device=device).unsqueeze(1).expand(-1, k)
    idx  = torch.stack([rows.reshape(-1), cols.reshape(-1)])
    vals = torch.randn(c * k, device=device)
    return torch.sparse_coo_tensor(idx, vals, (c, n)).coalesce().to_sparse_csr()

def timed_sparse_qk_cpu(C: int, D: int, k: int, alpha: int = 8):
    """
    CPU path: Q is CSR-sparse, Kᵀ is dense.
    Works on any PyTorch build (no MKL required).
    """
    D_prime = alpha * D
    torch.manual_seed(0)

    Q = _rand_k_sparse(C, D_prime, k)     # (C, D′)  CSR
    K = _rand_k_sparse(C, D_prime, k)     # (C, D′)  CSR  (will densify)

    K_T_dense = K.to_dense().t()          # (D′, C)  dense
    t0 = time.time()
    out = torch.sparse.mm(Q, K_T_dense)   # (C, C)  dense result
    print(f"time sparse×dense QKᵀ (CPU) with C={C} D'={D_prime} k={k}: {time.time()-t0:.6f}s")
    return out


def main():
    C = 4096
    D = 512
    timed_qk(C, D)
    timed_sparse_qk_cpu(C, D, k=64, alpha=8)

if __name__ == "__main__":
    main()