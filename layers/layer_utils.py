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

# Convention: any auxiliary loss exposed by layers should use this suffix in the aux dict.
# The model will aggregate keys ending with this suffix and weight them via config.
AUX_LOSS_SUFFIX = "layer_aux_loss"

class UnitColNormPadded(nn.Module):
    """
    Like UnitColNorm, but named explicitly for the 'padded D+1' use case.
    Normalizes along the specified dimension so that each vector has unit L2 norm.
    When used with an extra dummy coordinate, taking only the first D coordinates
    yields vectors with norm <= 1.
    """
    def __init__(self, dim=0, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, X):
        n = torch.linalg.norm(X, dim=self.dim, keepdim=True).clamp_min(self.eps)
        return X / n

    def right_inverse(self, Y):
        return Y
