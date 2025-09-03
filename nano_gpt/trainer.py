import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import Dict, List

from nano_gpt.data_loader import DataLoader
from nano_gpt.generator import Generator

class Trainer:
    # TODO - fix type of data_loader
    def __init__(
            self, 
            model: nn.Module, 
            data_loader: DataLoader, 
            char_level_tokenize: bool,
            sample_prompts: List[str],
            sample_length: int = 512):
        self.model = model
        self.loader = data_loader
        self.char_level_tokenize = char_level_tokenize
        self.generator = Generator(self.model, self.loader, self.char_level_tokenize, sample_prompts)
        self.sample_length = sample_length

    def train(self, lr: float, batch_size: int, steps: int, print_every: int) -> None:
        print("In Trainer::train", flush=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        for i in range(steps):
            # sample a batch of data
            xb, yb = self.loader.get_batch('train')  # BxC, BxC
            if i == 0:
                print("Successfully got batch", flush=True)
                print(f"xb shape: {xb.shape}")
                print(f"yb shape: {yb.shape}")

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            if (i % print_every == 0):
                print(f"Logits on sample 0: {logits[0]}")
                print("Got loss", flush=True)
                self.print_sample(i, loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

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
    
    def print_sample(self, it: int|None = None, loss: float|None = None):
        if it is not None:
            print(f"At training iteration {it}.\nLoss: {loss}")
        print("\nPrinting sample...", flush=True)
        # TODO - make this compatible w the character-level model too
        # TODO - make generator case insensitive
        self.generator.generate_from_prompts(self.sample_length)
    