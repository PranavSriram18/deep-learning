import torch
import torch.nn as nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from typing import Tuple

from nano_gpt.chunked_attention import ChunkedAttention  # type: ignore
from nano_gpt.transformer_blocks import MLP, Block  # type: ignore

torch.manual_seed(1337)

class ChunkedTransformer(nn.Module):
    """
    We'll hardcode the following for now:
    Context length: 256
    Chunk size: 16
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int
    ):
        super().__init__()
        # hyperparams
        self.V = vocab_size
        self.D = embedding_dim
        self.C = 256  # context length
        self.M = 16  # chunk size
        self.k = self.C // self.M # num chunks (16)
        self.num_heads = num_heads
        self.ff_expansion = 2
        self.num_layers = num_layers
        self.p_dropout = 0.  # no dropout for now


        assert(self.C % self.M == 0, "chunk size must divide context length")
        assert(self.D % self.num_heads == 0, "num_heads must divide embedding dim")
        
        # define the model architecture
        self.pos_emb = nn.Embedding(self.C, self.D)
        self.tok_emb = nn.Embedding(self.V, self.D)

        # BxCxD -> BxkxD
        self.att1 = ChunkedAttention(
            C=self.C, M=self.M, D=self.D, num_heads=self.num_heads)
        
        # BxkxD -> BxkxD
        self.mlp1 = MLP(D=self.D, ff_expansion=self.ff_expansion)

        # BxkxD -> BxkxD
        self.regular_atts = nn.Sequential(*[Block(
            self.D, self.num_heads, self.k, self.ff_expansion, self.p_dropout
            ) for _ in range(self.num_layers - 1)])
        
        self.ln_f = nn.LayerNorm(self.D)
        # BxkxD -> BxkxV
        self.lm_head = nn.Linear(self.D, self.V)

    def forward(self, idx, targets=None):
        """
        idx: BxC tensor of ints
        targets: BxC tensor of ints

        For compatibility with regular transformer training,
        targets will be a right-shifted version of idx. We want to only
        predict the start of each chunk (2nd chunk onwards), which are end of
        each chunk in target, so we want entries of target that are M-1 mod M
        """
        B, C = idx.shape
        if targets is not None:
            assert (targets.shape == (B, self.C))

        # Handle input padding/truncation
        if C < self.C:
            idx = F.pad(idx, (self.C - C, 0))  # Left-pad - TODO
        elif C > self.C:
            idx = idx[:, :self.C]  # Truncate
        C = self.C

        tok_emb = self.tok_emb(idx)  # B x C x D
        pos_emb = self.pos_emb(torch.arange(C, device=idx.device))  # C x D
        x = tok_emb + pos_emb  # BxCxD
        x = self.att1(x)  # BxkxD
        x = self.mlp1(x)  # BxkxD
        x = self.regular_atts(x)  # BxkxD
        x = self.ln_f(x)
        logits = self.lm_head(x)  # BxkxV
        
        if targets is None:
            loss = None
        else:
            B, k, V = logits.shape
            logits_rs = logits.view(B*k, V)
            targets_rs = targets.view(B, k, self.M)  # BxC -> BxkxM
            sub_targets = targets_rs[:, :, -1]  # Bxk (taking last elem of each of k chunks)
            sub_targets_rs = sub_targets.view(B*k)
            loss = F.cross_entropy(logits_rs, sub_targets_rs)  # (B*k)xV, (B*k)
        return logits, loss  # BxkxV, (1)
    
    def generate(self, idx, max_new_tokens) -> torch.Tensor:
        # idx is (B, C) array of indices in the current context
        for _ in range(max_new_tokens):
            # Handle input padding/truncation
            B, C = idx.shape
            if C < self.C:
                idx_crop = F.pad(idx, (self.C - C, 0))  # Left-pad - TODO
            elif C > self.C:
                idx_crop = idx[:, :self.C]  # Truncate
            C = self.C

            # get the predictions
            logits, _ = self(idx_crop, None)  # (B, k, V), None

            # focus only on the last time step and sample
            logits = logits[:, -1, :]  # (B, V)
            probs = F.softmax(logits, dim=-1)  # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, C+1)
        return idx
        