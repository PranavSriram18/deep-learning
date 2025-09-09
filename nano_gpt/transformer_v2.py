# nano_gpt/transformer_model.py
from nano_gpt.sliding_window_attn import SlidingWindowBlock
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from typing import Optional, Tuple

from nano_gpt.basic_sparse_attention import BasicSparseTransformerBlock
from nano_gpt.model_config import V2ModelConfig

torch.manual_seed(1337)


class TransformerV2(nn.Module):
    def __init__(self, config: V2ModelConfig):
        print(f"Initializing TransformerV2 with config {config}")
        super().__init__()
        self.V  = config.vocab_size
        self.D  = config.embedding_dim
        self.C  = config.context_length
        self.Cwi = config.sliding_window_init
        self.Cwf = config.sliding_window_final

        self.num_regular_init       = config.num_regular_init
        self.num_sparse        = config.num_sparse_blocks
        self.num_regular_final = config.num_regular_final
        self.num_heads_regular = config.num_heads_regular
        self.num_heads_sparse  = config.num_heads_sparse

        self.ff_expansion = config.ff_expansion
        self.dropout      = config.dropout

        self.alpha  = config.alpha
        self.t      = config.t
        self.use_ste = config.use_ste

        self.Dv = config.vocab_embed_dim  # factorized vocab bottleneck

        # Embedding: V -> Dv -> D
        self.token_embedding_table = nn.Embedding(self.V, self.Dv)
        self.embed_proj            = nn.Linear(self.Dv, self.D, bias=False)

        # Positional embeddings in D
        self.position_embedding_table = nn.Embedding(self.C, self.D)

        # Early: sliding-window (local) attention blocks with window Cwi
        self.sliding_window_stack_init = nn.Sequential(*[
            SlidingWindowBlock(
                D=self.D,
                num_heads=self.num_heads_regular,
                Cw=self.Cwi,
                ff_expansion=self.ff_expansion,
            )
            for _ in range(self.num_regular_init)
        ])

        # Later: sparse attention blocks over full context C
        self.sparse_stack = nn.Sequential(*[
            BasicSparseTransformerBlock(
                D=self.D,
                num_heads=self.num_heads_sparse,  # keyword name expected by this block
                C=self.C,
                alpha=self.alpha,
                t=self.t,
                ff_expansion=self.ff_expansion,
                dropout=self.dropout,
                use_ste=self.use_ste,
            )
            for _ in range(self.num_sparse)
        ])

        # finally: another sliding window stack
        self.sliding_window_stack_final = nn.Sequential(*[
            SlidingWindowBlock(
                D=self.D,
                num_heads=self.num_heads_regular,
                Cw=self.Cwf,
                ff_expansion=self.ff_expansion,
            )
            for _ in range(self.num_regular_final)
        ])

        self.ln_f = nn.LayerNorm(self.D)

        # Unembedding: D -> Dv -> V
        self.unembed_proj = nn.Linear(self.D, self.Dv, bias=False)
        self.lm_head      = nn.Linear(self.Dv, self.V, bias=False)

    def forward(
        self,
        idx: torch.Tensor,                 # (B, T) long
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        assert idx.dtype == torch.long, "idx must be Long for embedding"
        B, T = idx.shape
        dev = idx.device
        assert T <= self.C, "sequence length exceeds model context length"

        # Embed
        tok_dv = self.token_embedding_table(idx)   # (B, T, Dv)
        tok    = self.embed_proj(tok_dv)           # (B, T, D)

        # Add positional embeddings
        pos = torch.arange(T, device=dev)
        pos_emb = self.position_embedding_table(pos)     # (T, D)
        x = tok + pos_emb.unsqueeze(0)                   # (B, T, D)

        # Sliding-window then sparse stacks (both preserve sequence length)
        x = self.sliding_window_stack_init(x)                 # (B, T, D)
        x = self.sparse_stack(x)                         # (B, T, D)
        x = self.sliding_window_stack_final(x)                 # (B, T, D)

        # Final norm and unembedding
        x = self.ln_f(x)                                 # (B, T, D)
        y = self.unembed_proj(x)                         # (B, T, Dv)
        logits = self.lm_head(y)                         # (B, T, V)

        if targets is None:
            return logits, None

        assert targets.shape == idx.shape
        loss = F.cross_entropy(logits.view(B * T, self.V), targets.view(B * T))
        return logits, loss

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
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
