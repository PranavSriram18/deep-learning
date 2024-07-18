import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.nn import functional as F  # type: ignore

torch.manual_seed(1337)

class BigramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # each token reads off logits for next token from lookup table
        # V x V table; (i, j) entry is logit for token j following token i
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx, targets both (B, T) tensor of ints
        # For each token in idx, we pull the corresp. row from embedding table
        logits = self.token_embedding_table(idx)  # (B, T, V)
        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            # F.cross_entropy takes multiple shape combinations; we use
            # the (N x V) logits, (N) targets version
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions. __call__ internally calls forward and does
            # some internal bookkeepping for backprop
            logits, loss = self(idx)  # logits is (B, T, V)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, V)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, V)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # cat[(B, T), (B, 1)] -> (B, T+1)
        return idx
