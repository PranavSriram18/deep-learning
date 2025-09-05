# nano_gpt/transformer_model.py
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from typing import Optional, Tuple

from nano_gpt.transformer_blocks import Block
from nano_gpt.basic_sparse_attention import BasicSparseTransformerBlock
from nano_gpt.model_config import ModelConfig, TransformerType

torch.manual_seed(1337)


class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.V = config.vocab_size
        self.D = config.embedding_dim
        self.C = config.context_length
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.ff_expansion = config.ff_expansion
        self.dropout = config.dropout
        self.transformer_type = config.transformer_type
        self.alpha = config.alpha
        self.t = config.t
        self.use_ste = config.use_ste

        # Factorized embeddings: V -> Dv -> D  (and D -> Dv -> V for unembed)
        self.use_factorized_embeddings = getattr(config, "use_factorized_embeddings", False)
        self.Dv = int(getattr(config, "vocab_embed_dim", max(1, min(self.D, 256))))  # default small Dv if enabled
        tie = getattr(config, "tie_embeddings", True)

        if self.use_factorized_embeddings:
            # Input side
            self.token_embedding_table = nn.Embedding(self.V, self.Dv)          # V x Dv
            self.embed_proj = nn.Linear(self.Dv, self.D, bias=False)             # Dv -> D

            # Output side
            self.unembed_proj = nn.Linear(self.D, self.Dv, bias=False)           # D -> Dv
            if tie:
                # Tied logits: (B,T,Dv) @ (V,Dv)^T
                self.lm_head = None  # computed via matmul in forward
            else:
                self.lm_head = nn.Linear(self.Dv, self.V, bias=False)            # separate output matrix
        else:
            # Classic path: V -> D and D -> V (with optional tying at D)
            self.token_embedding_table = nn.Embedding(self.V, self.D)            # V x D
            self.embed_proj = None
            self.unembed_proj = None
            self.lm_head = nn.Linear(self.D, self.V, bias=False)
            if tie:
                if self.lm_head.weight.shape != self.token_embedding_table.weight.shape:
                    raise ValueError(
                        f"Cannot tie embeddings: head {tuple(self.lm_head.weight.shape)} "
                        f"!= embed {tuple(self.token_embedding_table.weight.shape)}"
                    )
                self.lm_head.weight = self.token_embedding_table.weight

        # Positions always in model dim D
        self.position_embedding_table = nn.Embedding(self.C, self.D)

        # Blocks
        if self.transformer_type == TransformerType.BASIC:
            self.blocks = nn.Sequential(
                *[Block(self.D, self.num_heads, self.C, self.ff_expansion, self.dropout)
                  for _ in range(self.num_layers)]
            )
        elif self.transformer_type == TransformerType.BASIC_SPARSE_ATTENTION:
            self.blocks = nn.Sequential(
                *[BasicSparseTransformerBlock(
                    self.D, self.num_heads, self.C, self.alpha, self.t, self.ff_expansion, self.dropout, self.use_ste)
                  for _ in range(self.num_layers)]
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.ln_f = nn.LayerNorm(self.D)

    def forward(
        self,
        idx: torch.Tensor,                 # (B, T) long
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        assert idx.dtype == torch.long, "idx must be Long for embedding"
        B, T = idx.shape
        dev = idx.device

        # Token embeddings (factorized or classic)
        if self.use_factorized_embeddings:
            tok_emb_dv = self.token_embedding_table(idx)             # (B, T, Dv)
            tok_emb = self.embed_proj(tok_emb_dv)                    # (B, T, D)
        else:
            tok_emb = self.token_embedding_table(idx)                # (B, T, D)

        # Positional embeddings in D
        pos = torch.arange(T, device=dev)
        pos_emb = self.position_embedding_table(pos)                 # (T, D)
        x = tok_emb + pos_emb.unsqueeze(0)                           # (B, T, D)

        x = self.blocks(x)                                           # (B, T, D)
        x = self.ln_f(x)                                             # (B, T, D)

        # Unembedding (factorized or classic)
        if self.use_factorized_embeddings:
            y = self.unembed_proj(x)                                 # (B, T, Dv)
            if self.lm_head is None:
                # Tied in Dv space: logits = y @ E^T
                logits = y @ self.token_embedding_table.weight.T     # (B, T, V)
            else:
                logits = self.lm_head(y)                             # (B, T, V)
        else:
            logits = self.lm_head(x)                                 # (B, T, V)

        if targets is None:
            return logits, None

        assert targets.shape == idx.shape
        loss = F.cross_entropy(logits.view(B * T, self.V), targets.view(B * T))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        dev = next(self.parameters()).device
        idx = idx.to(dev)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.C:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
