import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn


class ParallelTransformerBlock(nn.Module):
    def __init__(self, d_model=1024, d_head=64):
        super().__init__()
        assert (
            d_model % d_head == 0
        ), "Model dimension (`d_model`) but be a multiple of head dimension (`d_head`)"

        self.d_model = d_model
        self.d_head = d_head

        self.n_heads = d_model // d_head
        self.scaling_factor = d_head**-0.5

        ff_inner_dim = d_model * 4
        self.fused_dims = (d_model, d_head, d_head, 2 * ff_inner_dim)

        self.fused_attn_ff_proj = nn.Linear(d_model, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(), nn.Linear(ff_inner_dim, d_model, bias=False)
        )

        self.norm = LayerNorm(d_model)

        self.register_buffer("mask", None, persistent=False)

    def get_mask(self, seq_len):
        if self.mask is not None and self.mask.shape[-1] >= seq_len:
            return self.mask[:seq_len, :seq_len]

        mask = torch.ones((seq_len, seq_len), dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(self, x):
        assert x.dim() == 3, "3D tensor must be provided"

        # shape is B x C x M
        seq_len = x.shape[1]

        x = self.norm(x)

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        q = rearrange(
            q,
            "batch_size seq_len (n_heads d_head) -> batch_size n_heads seq_len d_head",
            n_heads=self.n_heads,
        )
        q = q * self.scaling_factor

        # b: batch_size, h: n_heads, d: d_head, t: seq_len_tgt, s: seq_len_src
        sim = einsum("b h t d, b s d -> b h t s", q, k)

        causal_mask = self.get_mask(seq_len)
        sim = sim.masked_fill(causal_mask, -torch.inf)

        attn = sim.softmax(dim=-1)

        # b: batch_size, h: n_heads, d: d_head, t: seq_len_tgt, s: seq_len_src
        out = einsum("b h t s, b s d -> b h t d", attn, v)
        out = rearrange(
            out,
            "batch_size n_heads seq_len d_head -> batch_size seq_len (n_heads d_head)",
        )

        return self.attn_out(out) + self.ff_out(ff)
