from abc import ABC, abstractmethod
from typing import Optional, TypeVar
import torch  # type: ignore

class BaseSparseRecoverySolver(ABC):
    def __init__(
        self,
        D: torch.Tensor, 
        y: torch.Tensor, 
        t: int, 
        mu: float = 1.0,
        eps: float = 1e-5,
        max_iters: int = 100,
    ):
        """
        Initialize solver instance.
        Params:
            D: Dictionary, shape d x n with n >> d. n columns are "atoms" of dimension d. Each is
            unit norm.
            y: target signal to reconstruct.
            t: max L0 norm of output x, i.e. number of nonzero coefficients.
            mu: step size.
            eps: convergence threshold.
            max_iters: maximum update iterations.

        Solver goal is to compute x in R^n that (approximately) minimizes ||Dx - y|| subject to
        L0(x) <= t.
        """
        self.d, self.n = D.shape
        self.D = D
        self.y = y
        self.t = t
        self.mu = mu
        self.max_iters = max_iters
        self.eps = eps

    @abstractmethod
    def solve(self, print_every: int = 10) -> torch.Tensor:
        """
        Output:
            x: vector that (approximately) minimizes ||Dx - y|| subject to L0(x) <= t.
        """
        pass

class IHTSolver(BaseSparseRecoverySolver):
    def __init__(
        self,
        D: torch.Tensor, 
        y: torch.Tensor, 
        t: int, 
        mu: float = 1.0,
        eps: float = 1e-5,
        max_iters: int = 100
    ):
        super().__init__(D, y, t, mu, eps, max_iters)

    def solve(self, print_every: int = 10) -> torch.Tensor:
        self._init()  # set initial x, residual, loss

        with torch.no_grad():
            it = 0
            # note: for initial experiments, just run all iters; later can add stopping criterion
            while it < self.max_iters:
                if it % print_every == 0:
                    print(f"\nIteration {it}:\n{self.x=}\ncurr_idxs={self.curr_idxs}\nloss={self.loss.item()} \n")
                self._update()
                it += 1
            print(f"Final result:\n{self.x=}\nloss={self.loss.item()}")

        return self.x

    def _init(self):
        self.x: torch.Tensor = torch.zeros(self.n, dtype=self.D.dtype, device=self.D.device)
        self.curr_idxs = torch.tensor([], dtype=torch.long, device=self.D.device)
        self.residual: torch.Tensor = self.y
        self.loss: torch.Tensor = torch.dot(self.residual, self.residual)

    def _update(self):
        """
        pre and post condition: self.x is t-sparse.
        """
        
        # get per-atom scores based on current residual
        scores = self.D.T @ self.residual  # (n, d) @ (d) -> (n)

        # mu-step (dense update on sparse iterate)
        updated_x = self.x + self.mu * scores  # (n)

        # hard threshold
        _, new_idxs = torch.topk(torch.abs(updated_x), k=self.t)
        self.x = torch.zeros_like(self.x)  # (n)
        self.x.scatter_(0, new_idxs, updated_x.index_select(0, new_idxs))  # (n), t-sparse
        self.curr_idxs = new_idxs

        # new residual and loss
        self.residual = self.y - self.D @ self.x  # (d)
        self.loss = torch.dot(self.residual, self.residual)  # scalar


# Testing


T = TypeVar("T", bound=BaseSparseRecoverySolver)

def test_solver(SolverCls: type[T]) -> None:
    # target signal is in R^d, approx equal to sum of t of n atoms
    d, n, t = 5, 10, 3

    # build column-normalized dict. random-vectors are ~RIP
    D = torch.randn(d, n, dtype=torch.float64)  # (d, n)
    col_norms = torch.linalg.norm(D, dim=0) + 1e-12  # (n,)
    D = D / col_norms

    L_est = torch.linalg.norm(D, 2).item() ** 2
    print(f"Built dictionary.\nFirst atom: {D[:, 0]}\n{L_est=}")

    # build target
    mask = torch.zeros(n, dtype=D.dtype, device=D.device)
    indices = torch.randperm(n)[:t]
    mask.scatter_(0, indices, 1)  # place 1s at selected indices
    x_true = torch.randn(n, dtype=D.dtype, device=D.device) * mask  # (n,) t-sparse
    noise_coeff = 1e-10
    y = D @ x_true + torch.randn(d, dtype=D.dtype, device=D.device) * noise_coeff  # (d,)

    # hyperparams
    mu = 1.0 / (2.0 * L_est)
    eps, max_iters, print_every = 1e-5, 250, 10
    torch.set_printoptions(precision=3, sci_mode=False)

    solver = SolverCls(
        D, y, t, mu, eps, max_iters
    )

    print(f"Running test with x_true={x_true}")

    solver.solve(print_every)


if __name__=="__main__":
    test_solver(IHTSolver)

