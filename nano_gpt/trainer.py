import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import Dict

from bigram_model import BigramModel
from data_loader import DataLoader

class Trainer:
    def __init__(self, model: nn.Module, data_loader: DataLoader):
        self.model = model
        self.loader = data_loader

    def train(self, lr: float, batch_size: int, steps: int, print_every: int) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        for i in range(steps):
            if (i % print_every == 0):
                self.print_sample(i)
            # sample a batch of data
            xb, yb = self.loader.get_batch('train')

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(loss.item())

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int) -> Dict[str, float]:
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.loader.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def print_sample(self, it: int = None):
        s = "\nPrinting sample"
        s += f" at train iter {it}" if it else ":"
        print(s, flush=True)
        encoded = self.model.generate(
            idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=50)[0].tolist()
        print(self.loader.decode(encoded), flush=True)
    