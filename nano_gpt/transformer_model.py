import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

from nano_gpt.transformer_blocks import Block

torch.manual_seed(1337)

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        num_heads: int,
        num_layers: int,
        ff_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.V = vocab_size
        self.D = embedding_dim
        self.C = context_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_expansion = ff_expansion
        self.dropout = dropout

        self.token_embedding_table = nn.Embedding(self.V, self.D)
        self.position_embedding_table = nn.Embedding(self.C, self.D)

        self.blocks = nn.Sequential(
            *[Block(self.D, self.num_heads, self.C, self.ff_expansion, self.dropout)
              for _ in range(self.num_layers)]
        )
        self.ln_f = nn.LayerNorm(self.D)
        self.lm_head = nn.Linear(self.D, self.V)


    def forward(
        self,
        idx: torch.Tensor,                 # (B, T) long
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        assert idx.dtype == torch.long, "idx must be Long for embedding"
        B, T = idx.shape

        # device from data; works regardless of where the module lives
        dev = idx.device

        tok_emb = self.token_embedding_table(idx)  # (B, T, D)

        # build positional indices on the SAME device as idx
        pos = torch.arange(T, device=dev)          # (T,)
        pos_emb = self.position_embedding_table(pos)  # (T, D)
        x = tok_emb + pos_emb.unsqueeze(0)            # (B, T, D)

        x = self.blocks(x)            # (B, T, D)
        x = self.ln_f(x)              # (B, T, D)
        logits = self.lm_head(x)      # (B, T, V)

        loss: Optional[torch.Tensor]
        if targets is None:
            loss = None
        else:
            assert targets.shape == idx.shape
            loss = F.cross_entropy(
                logits.view(B * T, self.V),
                targets.reshape(B * T)
            )
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # Make generation robust to caller device
        dev = next(self.parameters()).device
        idx = idx.to(dev)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.C:]         # (B, T<=C)
            logits, _ = self(idx_cond)          # (B, T, V)
            logits = logits[:, -1, :]           # (B, V)
            probs = F.softmax(logits, dim=-1)   # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx
