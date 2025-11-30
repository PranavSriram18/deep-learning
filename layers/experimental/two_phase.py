import torch
import torch.nn as nn
import math

class TwoPhaseLift(nn.Module):
    """
    Two-phase (D1 -> H -> D2) linear map with contiguous cycling connectivity.

    Notation:
      D1, H, D2 : layer widths
      B1, B2    : block sizes (contiguous fan-in per neuron in stage 1 and stage 2)
      G1        : number of input blocks in stage 1  = D1 // B1
      GH2       : number of hidden blocks in stage 2 = H  // B2   (naming avoids confusion)

    Stage 1 (D1 -> H), cycling rule:
      - Split D1 into G1 contiguous blocks of size B1.
      - H is partitioned into groups of size G1 (so there are H//G1 groups).
      - For each j in {0..G1-1}, the j-th neuron within *every* H-group connects
        to the j-th input block (same block index), i.e., source block = (h_index % G1).
      - Each such connection uses its own learned weights over the B1 inputs.
      - Implementation: batched GEMM per input block, then INTERLEAVE groups across H.

    Stage 2 (H -> D2), contiguous rule:
      - Split H into GH2 contiguous blocks of size B2.
      - D2 is partitioned into GH2 groups (so each group size is D2pb = D2 // GH2).
      - Each D2 neuron connects to exactly one H block (contiguous B2 fan-in).
      - Implementation: batched GEMM per hidden block.

    IMPORTANT coverage note:
      - With the stage-1 cycling, a contiguous window of length B2 over H will
        cover *all* G1 input blocks iff B2 >= G1.
      - If B2 < G1, each D2 neuron only "sees" a subset of D1 blocks.
        (You can change the stage-2 pattern to strided blocks to cover all with smaller B2.)
    """
    def __init__(self, D1, H, D2, B1, B2, dtype=torch.float32, device=None):
        super().__init__()
        assert D1 % B1 == 0, "D1 must be divisible by B1"
        assert H  % B2 == 0, "H must be divisible by B2"
        self.D1, self.H, self.D2 = D1, H, D2
        self.B1, self.B2 = B1, B2

        # Groups/block counts per your notation
        self.G1  = D1 // B1          # number of D1 blocks / H-group size
        self.GH2 = H  // B2          # number of H blocks used in stage 2
        assert H % self.G1 == 0, "H must be a multiple of G1 so H groups tile cleanly"
        assert D2 % self.GH2 == 0, "D2 must be divisible by number of H blocks"
        self.H_per_block1 = H  // self.G1     # columns of H per *input* block (pre-interleave)
        self.D2_per_block2 = D2 // self.GH2   # columns of D2 per *hidden* block

        # Parameters stored as compact batched dense weights (hardware friendly)
        k1 = 1.0 / math.sqrt(self.B1)
        k2 = 1.0 / math.sqrt(self.B2)
        # Stage 1: one (B1 -> H_per_block1) weight per input block
        self.W1 = nn.Parameter(torch.empty(self.G1, self.B1, self.H_per_block1,
                                           dtype=dtype, device=device).uniform_(-k1, k1))
        # Stage 2: one (B2 -> D2_per_block2) weight per hidden block
        self.W2 = nn.Parameter(torch.empty(self.GH2, self.B2, self.D2_per_block2,
                                           dtype=dtype, device=device).uniform_(-k2, k2))

    def forward(self, x):
        """
        x: (C, D1)  ->  y: (C, D2)
        """
        C = x.shape[0]

        # ---- Stage 1: D1 -> H (cycling across input blocks)
        # 1) Slice x into G1 contiguous blocks of size B1
        #    xb: (C, G1, B1)
        xb = x.view(C, self.G1, self.B1).contiguous()

        # 2) Batched block GEMM: for each input block g, (C, B1) @ (B1, H_per_block1)
        #    gives (C, H_per_block1). Stack over g -> (C, G1, H_per_block1).
        y1_blocked = torch.einsum('cgb,gbh->cgh', xb, self.W1)  # (C, G1, H_per_block1)

        # 3) INTERLEAVE across H so that column h uses source block (h % G1).
        #    We do: (C, G1, Hpb) -> (C, Hpb, G1) -> reshape to (C, H)
        y1 = y1_blocked.permute(0, 2, 1).contiguous().view(C, self.H)  # (C, H)

        # ---- Stage 2: H -> D2 (contiguous blocks)
        # 4) Slice y1 into GH2 contiguous hidden blocks of size B2
        y1b = y1.view(C, self.GH2, self.B2).contiguous()               # (C, GH2, B2)

        # 5) Batched block GEMM per hidden block: (C, B2) @ (B2, D2pb)
        y2_blocked = torch.einsum('cgb,gbd->cgd', y1b, self.W2)        # (C, GH2, D2pb)

        # 6) Concatenate D2 groups
        y = y2_blocked.reshape(C, self.D2)                              # (C, D2)
        return y

    @torch.no_grad()
    def effective_dense(self):
        """
        Materialize the explicit dense matrix Ŵ ∈ R^{D1×D2} represented by the two-phase lift.
        Handy for sanity checks at small sizes.
        """
        dev, dt = self.W1.device, self.W1.dtype
        # Build the stage-1 dense map with cycling:
        # Start with blockwise map (D1 x H) then interleave H columns.
        W1_block = torch.zeros(self.D1, self.H, dtype=dt, device=dev)
        for g in range(self.G1):
            r0, r1 = g*self.B1, (g+1)*self.B1
            c0, c1 = g*self.H_per_block1, (g+1)*self.H_per_block1
            W1_block[r0:r1, c0:c1] = self.W1[g]  # (B1, Hpb)

        # interleave: (G1 blocks of Hpb columns) -> cycle by taking columns in order
        # col index h picks from block (h % G1), offset h // G1
        W1_full = torch.empty(self.D1, self.H, dtype=dt, device=dev)
        Hpb = self.H_per_block1
        for h in range(self.H):
            g = h % self.G1
            off = h // self.G1
            W1_full[:, h] = W1_block[:, g*Hpb + off]

        # Build stage-2 dense map (block-diagonal over H blocks)
        W2_full = torch.zeros(self.H, self.D2, dtype=dt, device=dev)
        for g in range(self.GH2):
            r0, r1 = g*self.B2, (g+1)*self.B2
            c0, c1 = g*self.D2_per_block2, (g+1)*self.D2_per_block2
            W2_full[r0:r1, c0:c1] = self.W2[g]  # (B2, D2pb)

        return W1_full @ W2_full  # (D1, D2)

