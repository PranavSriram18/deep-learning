import torch  # type: ignore
from torch import nn  # type: ignore

class UnitColNorm(nn.Module):
    def __init__(self, dim=0, eps=1e-12):
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, X):
        # X is the original, unconstrained tensor
        n = torch.linalg.norm(X, dim=self.dim, keepdim=True).clamp_min(self.eps)
        return X / n

    def right_inverse(self, Y):
        # Given a constrained Y, return an original so that forward(original) == Y.
        # For normalization, identity works: forward(Y) == Y already.
        return Y
