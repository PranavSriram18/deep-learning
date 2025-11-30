# nano_gpt/sliding_window_attn.py
from layers.layer_config import AttentionConfig
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F  # type: ignore


class SlidingWindowHead(nn.Module):
    """
    Single head of local causal self-attention with window Cw.

    Shapes:
      X:   (B, T, D)        input embeddings
      out: (B, T, H)        head output

    Complexity:
      O(B * T * H * Cw) compute and memory. Builds per-token windows via unfold,
      producing attention scores of shape (B, T, Cw) instead of (B, T, T).
    """
    def __init__(self, H: int, D: int, Cw: int):
        super().__init__()
        print(f"Initializing SlidingWindowHead (new version): H={H}, D={D}, Cw={Cw}")
        self.key   = nn.Linear(D, H, bias=False)
        self.query = nn.Linear(D, H, bias=False)
        self.value = nn.Linear(D, H, bias=False)
        self.Cw = int(Cw)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, _ = X.shape
        H = self.key.out_features
        Cw = self.Cw
        pad = max(0, Cw - 1)

        Q = self.query(X)  # (B, T, H)
        K = self.key(X)    # (B, T, H)
        V = self.value(X)  # (B, T, H)

        # Left-pad time so each position i has keys from [i-Cw+1, ..., i]
        if pad > 0:
            Kp = F.pad(K, (0, 0, pad, 0, 0, 0))  # (B, T+pad, H)
            Vp = F.pad(V, (0, 0, pad, 0, 0, 0))  # (B, T+pad, H)
        else:
            Kp, Vp = K, V

        # Unfold adds window as the LAST dim: (B, T, H, Cw). Permute to (B, T, Cw, H).
        Kwin = Kp.unfold(dimension=1, size=Cw, step=1).permute(0, 1, 3, 2)
        Vwin = Vp.unfold(dimension=1, size=Cw, step=1).permute(0, 1, 3, 2)

        # Scores over the window: (B, T, Cw)
        att = (Q.unsqueeze(2) @ Kwin.transpose(-1, -2)).squeeze(2) / (H ** 0.5)

        # Mask padded slots for the first positions
        if pad > 0:
            i = torch.arange(T, device=X.device).unsqueeze(1)  # (T,1)
            j = torch.arange(Cw, device=X.device).unsqueeze(0) # (1,Cw)
            valid = j <= i                                     # (T,Cw)
            att = att.masked_fill(~valid.unsqueeze(0), float('-inf'))

        att = F.softmax(att, dim=-1)           # (B, T, Cw)

        # Weighted sum of values: (B,T,1,Cw) @ (B,T,Cw,H) -> (B,T,1,H) -> (B,T,H)
        out = (att.unsqueeze(2) @ Vwin).squeeze(2)
        return out

class SlidingWindowAttention(nn.Module):
    """
    Multi-head local causal attention with window Cw.

    Shapes:
      X:   (B, T, D)
      out: (B, T, D)
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()
        D, C, num_heads = config.D, config.C, config.num_heads
        assert D % num_heads == 0, "Embedding dim D must be divisible by num_heads"
        H = D // num_heads
        self.heads = nn.ModuleList(
            [SlidingWindowHead(H=H, D=D, Cw=C) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(D, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Concatenate head outputs and project back to model dim
        head_outputs = torch.cat([h(X) for h in self.heads], dim=-1)  # (B, T, D)
        out = self.proj(head_outputs)                  # (B, T, D)
        return X + out
