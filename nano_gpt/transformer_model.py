import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer_blocks import Block

torch.manual_seed(1337)

class TransformerModel(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            embedding_dim, 
            context_length,
            num_heads,
            num_layers,
            dropout,
            device
        ):
        super().__init__()
        # hyperparams
        self.V = vocab_size
        self.D = embedding_dim
        self.C = context_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        # Each row is embedding for a token
        self.token_embedding_table = nn.Embedding(self.V, self.D)
        # each row is embedding for a pos
        self.position_embedding_table = nn.Embedding(self.C, self.D)  
        self.blocks = nn.Sequential(
            *[Block(self.D, self.num_heads, self.C, self.dropout) for _ in range(
                self.num_layers)])
        self.ln_f = nn.LayerNorm(self.D) # final layer norm
        self.lm_head = nn.Linear(self.D, vocab_size)  # token selection

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last context length tokens
            idx_cond = idx[:, -self.C:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx