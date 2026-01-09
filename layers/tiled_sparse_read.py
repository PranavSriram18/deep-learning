import torch  # type: ignore
from torch import nn  # type: ignore
import torch.nn.utils.parametrize as parametrize  # type: ignore

from layers.layer_utils import UnitColNorm


class TiledSparseRead(nn.Module):
    """
    Two-way tiling sparse read.

    Assumptions:
      - b is power of 2 in [8, 64]
      - D divisible by 2*b
      - m divisible by 2*T where T = D/(2*b)
      - D divisible by 4 (implied by b>=8 and D divisible by 2*b in most setups, but checked anyway)

    Construction:
      T = D/(2*b)
      Horizontal tiles: view x as (D/4, 4), take T row-slices of size (b/2, 4)
      Vertical tiles:   view x as (4, D/4), take T col-slices of size (4, b/2)
      Total groups G = 2*T, each tile corresponds to a group.
    """

    def __init__(self, D: int, m: int, b: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.D = D
        self.m = m
        self.b = b
        self.dtype = dtype

        if b & (b - 1) != 0 or not (8 <= b <= 64):
            raise ValueError(f"b must be a power of 2 in [8,64], got b={b}")
        if D % (2 * b) != 0:
            raise ValueError(f"D must be divisible by 2*b, got D={D}, b={b}")
        if D % 4 != 0:
            raise ValueError(f"D must be divisible by 4, got D={D}")

        self.T = D // (2 * b)
        self.G = 2 * self.T
        if m % self.G != 0:
            raise ValueError(f"m must be divisible by 2*T={self.G}, got m={m}")

        self.experts_per_group = m // self.G
        self.tile_size = 2 * b

        idx = self._build_tile_indices(D=D, b=b)  # (G, tile_size)
        self.register_buffer("tile_idx", idx, persistent=True)

        # V: (G, experts_per_group, tile_size, b)
        self.V = nn.Parameter(
            torch.randn(self.G, self.experts_per_group, self.tile_size, self.b, dtype=self.dtype)
        )
        # Normalize each neuron's incoming weight vector over tile_size
        parametrize.register_parametrization(self, "V", UnitColNorm(dim=2))

    @staticmethod
    def _build_tile_indices(D: int, b: int) -> torch.Tensor:
        T = D // (2 * b)
        d4 = D // 4
        half_b = b // 2

        # Horizontal tiles from view (D/4, 4): idx = r*4 + c
        horiz = []
        for t in range(T):
            r0 = t * half_b
            r1 = (t + 1) * half_b
            rows = torch.arange(r0, r1, dtype=torch.long)  # (b/2,)
            cols = torch.arange(0, 4, dtype=torch.long)    # (4,)
            rr = rows[:, None].expand(half_b, 4)
            cc = cols[None, :].expand(half_b, 4)
            tile = (rr * 4 + cc).reshape(-1)  # (2*b,)
            horiz.append(tile)
        horiz_idx = torch.stack(horiz, dim=0)  # (T, 2*b)

        # Vertical tiles from view (4, D/4): idx = r*(D/4) + c
        vert = []
        for t in range(T):
            c0 = t * half_b
            c1 = (t + 1) * half_b
            cols = torch.arange(c0, c1, dtype=torch.long)  # (b/2,)
            rows = torch.arange(0, 4, dtype=torch.long)    # (4,)
            rr = rows[:, None].expand(4, half_b)
            cc = cols[None, :].expand(4, half_b)
            tile = (rr * d4 + cc).reshape(-1)  # (2*b,)
            vert.append(tile)
        vert_idx = torch.stack(vert, dim=0)  # (T, 2*b)

        return torch.cat([horiz_idx, vert_idx], dim=0)  # (2*T, 2*b)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: (N, D)
        returns h_all: (N, m, b)
        """
        N, D = x_flat.shape
        if D != self.D:
            raise ValueError(f"Expected D={self.D}, got D={D}")

        # x_tiles: (N, G, tile_size)
        # Advanced indexing with a 2D index tensor yields (N, G, tile_size).
        # Note: this is the least gpu friendly part of this kernel
        x_tiles = x_flat[:, self.tile_idx]

        # (N,G,tile) @ (G,e,tile,b) -> (N,G,e,b)
        h = torch.einsum("ngt,getb->ngeb", x_tiles, self.V)

        # Flatten groups -> experts: (N, m, b)
        return h.reshape(N, self.m, self.b)
