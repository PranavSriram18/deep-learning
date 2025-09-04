import torch
import torch.nn as nn
from typing import Dict, List, Optional

from nano_gpt.data_loader import ShakespeareDataLoader
from nano_gpt.data_wt2_word import WT2WordDataLoader

from nano_gpt.generator import Generator

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data_loader: ShakespeareDataLoader | WT2WordDataLoader,  # TODO - make a base class for this
        char_level_tokenize: bool,
        sample_prompts: List[str],
        sample_length: int = 512,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ):
        # Resolve and store device
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.loader = data_loader
        self.char_level_tokenize = char_level_tokenize
        self.sample_length = sample_length
        self.use_amp = use_amp and (self.device.type == "cuda")

        # Generator should already work if your model.generate is device-aware,
        # but we keep a handle here anyway.
        self.generator = Generator(self.model, self.loader, self.char_level_tokenize, sample_prompts)

        # AMP scaler (no-op when enabled=False)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

    def train(self, lr: float, batch_size: int, steps: int, print_every: int) -> None:
        print("In Trainer::train", flush=True)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for i in range(steps):
            # ---- sample a batch
            xb, yb = self.loader.get_batch('train')   # shapes: (B, T), (B, T)
            # move to model device
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            if i == 0:
                print("Successfully got batch", flush=True)
                print(f"xb shape/dev: {xb.shape} / {xb.device}")
                print(f"yb shape/dev: {yb.shape} / {yb.device}")
                print(f"model dev: {next(self.model.parameters()).device}")

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits, loss = self.model(xb, yb)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits, loss = self.model(xb, yb)
                loss.backward()
                optimizer.step()

            if (i % print_every == 0):
                # Small, safe log (avoid printing giant tensors)
                print(f"step {i} | loss {loss.item():.4f} | logits[0,-1,:5]={logits[0, -1, :5].detach().cpu().tolist()}")
                self.print_sample(i, loss.item())

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)  # keep on CPU; we store .item() anyway
            for k in range(eval_iters):
                X, Y = self.loader.get_batch(split)
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        _, loss = self.model(X, Y)
                else:
                    _, loss = self.model(X, Y)
                losses[k] = float(loss.item())
            out[split] = float(losses.mean().item())
        self.model.train()
        return out

    def print_sample(self, it: int | None = None, loss: float | None = None):
        if it is not None:
            print(f"At training iteration {it}. Loss: {loss:.4f}")
        print("\nPrinting sample...", flush=True)
        # Generator calls model.generate(), which is device-aware
        outs = self.generator.generate_from_prompts(self.sample_length, display=False)
        is_word_level = not self.char_level_tokenize
        for out in outs:
            text = " ".join(out) if is_word_level else "".join(out)
            print(f"\n{text}\n", flush=True)
