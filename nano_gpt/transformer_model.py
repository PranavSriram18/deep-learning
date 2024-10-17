import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from typing import Tuple

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
        device: str = 'cpu'
    ):
        super().__init__()
        # hyperparams
        self.V = vocab_size
        self.D = embedding_dim
        self.C = context_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_expansion = ff_expansion
        self.dropout = dropout
        self.device = device

        # Maps tokens (R^V) to embeddings (R^D)
        self.token_embedding_table = nn.Embedding(self.V, self.D)
        # Maps positions (R^C) to embeddings (R^D)
        self.position_embedding_table = nn.Embedding(self.C, self.D)  
        self.blocks = nn.Sequential(
            *[Block(self.D, self.num_heads, self.C, 
                    self.ff_expansion, self.dropout) for _ in range(
                self.num_layers)])
        self.ln_f = nn.LayerNorm(self.D) # final layer norm
        self.lm_head = nn.Linear(self.D, self.V)  # token selection

    def forward(self, idx, targets=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # idx and targets are both (B,C) tensor of integers
        B, C = idx.shape
        if targets is not None:
            assert(targets.shape == idx.shape)  # TODO - temp
        tok_emb = self.token_embedding_table(idx)  # (B, C, D)
        pos_emb = self.position_embedding_table(torch.arange(C, device=self.device))  # (C, D)
        x = tok_emb + pos_emb  # (B, C, D) (broadcast position embs across batch)
        x = self.blocks(x)  # (B, C, D)
        x = self.ln_f(x)  # (B, C, D)
        logits = self.lm_head(x) # (B, C, V)

        if targets is None:
            loss = None
        else:
            B, C, V = logits.shape
            logits = logits.view(B*C, V)
            targets = targets.reshape(B*C)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last context length tokens
            idx_cond = idx[:, -self.C:]  # (B, C') where C' = min(C, T)
            # get the predictions
            logits, loss = self(idx_cond)  # (B, C', V)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, V)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, V)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, C'+1)
        return idx
    