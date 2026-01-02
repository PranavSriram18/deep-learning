import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from data.base_loader import BaseLoader, DataMode
from nano_gpt.generator import Generator

@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    print_every: int
    train_steps: int
    sample_length: int
    sample_prompts: list[str]
    char_level_tokenize: bool
    use_amp: bool
    checkpoint_path: Optional[str] = None
    checkpoint_every: int = 0  # 0 disables periodic checkpointing

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data_loader: BaseLoader,
        train_config: TrainConfig,
        device: Optional[torch.device] = None,
    ):
        # Resolve and store device
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.loader = data_loader
        self.char_level_tokenize = train_config.char_level_tokenize
        self.sample_length = train_config.sample_length
        self.use_amp = train_config.use_amp and (self.device.type == "cuda")

        self.generator = Generator(self.model, self.loader, self.char_level_tokenize, train_config.sample_prompts)

        # AMP scaler (no-op when enabled=False)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.checkpoint_path: Optional[str] = train_config.checkpoint_path
        self.checkpoint_every: int = int(train_config.checkpoint_every or 0)
        self._train_config = train_config

    def train(self, lr: float, batch_size: int, steps: int, print_every: int) -> None:
        print("In Trainer::train", flush=True)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # decay each time print_every steps elapse
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=print_every, gamma=0.985)

        for i in range(steps):
            # ---- sample a batch
            xb, yb = self.loader.get_batch(DataMode.TRAIN)   # shapes: (B, T), (B, T)
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
                    logits, loss, aux_loss = self.model(xb, yb)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits, loss, aux_loss = self.model(xb, yb)
                loss.backward()
                optimizer.step()

            self.scheduler.step()

            if (i % print_every == 0):
                print(f"step {i} | train loss {loss.item():.4f} (aux {aux_loss.item():.4f}) | logits[0,-1,:5]={logits[0, -1, :5].detach().cpu().tolist()}")
                # Also report evaluation loss (single batch) with aux breakdown
                with torch.no_grad():
                    Xv, Yv = self.loader.get_batch(DataMode.EVAL)
                    Xv = Xv.to(self.device, non_blocking=True)
                    Yv = Yv.to(self.device, non_blocking=True)
                    if self.use_amp:
                        with torch.amp.autocast(device_type="cuda"):
                            _, eval_loss, eval_aux = self.model(Xv, Yv)
                    else:
                        _, eval_loss, eval_aux = self.model(Xv, Yv)
                print(f"eval loss {eval_loss.item():.4f} (aux {eval_aux.item():.4f})")
                self.print_sample(i, loss.item())

            # periodic checkpointing independent of print cadence
            if self.checkpoint_every and (i % self.checkpoint_every == 0):
                self._maybe_save_checkpoint(loss, i, optimizer, silent=True)


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

    def _maybe_save_checkpoint(self, loss: torch.Tensor, i: int, optimizer: torch.optim.Optimizer, silent: bool = False):
        if not self.checkpoint_path:
            return
        extra_state_fn = getattr(self.model, "extra_checkpoint_state", None)
        extra: dict = {}
        if callable(extra_state_fn):
            try:
                extra = extra_state_fn() or {}
            except Exception:
                extra = {}
        payload: dict = {
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": getattr(self, "scheduler", None).state_dict() if hasattr(self, "scheduler") else None,
            "model_config": asdict(self.model.config) if hasattr(self.model, "config") else None,
            "train_config": asdict(self._train_config) if self._train_config else None,
            "step": int(i),
            "loss": float(loss.item()),
        }
        payload |= extra
        try:
            torch.save(payload, self.checkpoint_path)
            if not silent:
                print(f"Saved checkpoint to {self.checkpoint_path} at step {i}")
        except Exception as e:
            if not silent:
                print(f"Warning: failed to save checkpoint to {self.checkpoint_path}: {e}")
