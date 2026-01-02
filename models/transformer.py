# nano_gpt/transformer_model.py
from layers.transformer_block import TransformerBlock
from layers.layer_utils import AUX_LOSS_SUFFIX
from models.transformer_config import TransformerConfig
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from typing import Any, Optional, Tuple


torch.manual_seed(1337)

class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.V = config.vocab_size
        self.D = config.embedding_dim
        self.C = config.context_length
        self.num_layers = config.num_layers

        # Factorized embeddings: V -> Dv -> D  (and D -> Dv -> V for unembed)
        self.use_factorized_embeddings = config.use_factorized_embeddings
        self.Dv = config.vocab_embed_dim
        if not self.use_factorized_embeddings:
            assert self.Dv == self.D, "vocab embed dim must match embed dim if not using factorized embeddings"
        self.tie_embeddings = config.tie_embeddings
        
        # self.transformer_type = config.transformer_type
        self.config = config

        # Embeddings
        self.token_embedding_table = nn.Embedding(self.V, self.Dv)          # V x Dv
        self.lm_head = nn.Linear(self.Dv, self.V, bias=False)  # Dv -> V
        if self.tie_embeddings:
            self.lm_head.weight = self.token_embedding_table.weight

        self.embed_proj, self.unembed_proj = None, None
        if self.use_factorized_embeddings:
            self.embed_proj = nn.Linear(self.Dv, self.D, bias=False)             # Dv -> D
            self.unembed_proj = nn.Linear(self.D, self.Dv, bias=False)           # D -> Dv

        # Positions always in model dim D
        self.position_embedding_table = nn.Embedding(self.C, self.D)

        # Blocks
        self.blocks = nn.ModuleList(
            TransformerBlock(config.block_config) for _ in range(self.num_layers)
        )
        self.ln_f = nn.LayerNorm(self.D)

    def extra_checkpoint_state(self) -> dict:
        """
        Models can override/extend this to save additional state alongside weights.
        Return a flat dict; it will be merged into the checkpoint payload.
        """
        # Example: include any relevant scalar attributes if present
        extra = {}
        if hasattr(self, "t"):
            try:
                extra["t"] = int(getattr(self, "t"))
            except Exception:
                pass
        return extra

    def forward(
        self,
        idx: torch.Tensor,                 # (B, T) long
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:

        assert idx.dtype == torch.long, "idx must be Long for embedding"
        B, T = idx.shape
        dev = idx.device

        # Token embeddings
        if self.use_factorized_embeddings:
            tok_emb_dv = self.token_embedding_table(idx)             # (B, T, Dv)
            tok_emb = self.embed_proj(tok_emb_dv)                    # (B, T, D)
        else:
            tok_emb = self.token_embedding_table(idx)                # (B, T, D)

        # Positional embeddings
        pos = torch.arange(T, device=dev)
        pos_emb = self.position_embedding_table(pos)                 # (T, D)
        x = tok_emb + pos_emb.unsqueeze(0)                           # (B, T, D)

        # Transformer blocks + final LN
        aux_dict: dict[str, Any] = {}
        for i, blk in enumerate(self.blocks):
            x, blk_dict = blk(x)  # (B, T, D)
            for k, v in blk_dict.items():
                aux_dict[f"block_{i}.{k}"] = v        
        x = self.ln_f(x)                                             # (B, T, D)

        # Unembedding
        if self.use_factorized_embeddings:
            y = self.unembed_proj(x)                                 # (B, T, Dv)
            logits = self.lm_head(y)                             # (B, T, V)
        else:
            logits = self.lm_head(x)                                 # (B, T, V)

        # Auxiliary loss from blocks (0 when disabled)
        aux = self.aux_loss(aux_dict)
        # Aggregate non-loss metrics across layers (mean over available layers)
        self.last_metrics = self.aggregate_metrics(aux_dict)
        if targets is None:
            return logits, None, aux

        assert targets.shape == idx.shape
        loss = F.cross_entropy(logits.view(B * T, self.V), targets.view(B * T))
        loss = loss + aux
        return logits, loss, aux

    def aux_loss(self, aux_dict: dict[str, Any]) -> torch.Tensor:
        # Aggregate auxiliary losses (e.g., from SparseExpertV3) with a configurable weight
        weight = getattr(self.config, "aux_loss_weight", 0.0)
        if weight == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        aux_terms = []
        for k, v in aux_dict.items():
            if k.endswith(AUX_LOSS_SUFFIX) and isinstance(v, torch.Tensor):
                aux_terms.append(v)
        if not aux_terms:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        # average auxiliary loss across blocks/layers to keep scale stable
        aux_mean = torch.stack([t.to(next(self.parameters()).device) for t in aux_terms]).mean()
        return weight * aux_mean

    def aggregate_metrics(self, aux_dict: dict[str, Any]) -> dict[str, float]:
        keys = ["num_low_scoring_experts", "num_near_dead_experts"]
        out: dict[str, float] = {}
        for metric in keys:
            vals = []
            for k, v in aux_dict.items():
                if k.endswith(metric) and isinstance(v, torch.Tensor):
                    vals.append(v.to(next(self.parameters()).device).float())
            if vals:
                out[metric] = torch.stack(vals).mean().item()
        return out

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, greedy: bool = False
    ) -> torch.Tensor:
        """
        Generate continuation for a single prompt.
        TODO: add support for greedy
        """
        dev = next(self.parameters()).device
        idx = idx.to(dev)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.C:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
