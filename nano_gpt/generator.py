import torch
import torch.nn as nn
from typing import List, Optional

from nano_gpt.data_loader import DataLoader

class Generator:
    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        char_level_tokenize: bool,         # (renamed for clarity)
        sample_prompts: List[str],
    ):
        self.model = model
        self.loader = loader
        self.char_level_tokenize = char_level_tokenize
        self.prompts = sample_prompts

    # ----- helpers -----
    def _device(self) -> torch.device:
        # derive from model parameters — single source of truth
        return next(self.model.parameters()).device

    def _to_idx(self, prompt: str) -> torch.Tensor:
        if self.char_level_tokenize:
            ids = self.loader.encode(list(prompt))
        else:
            ids = self.loader.encode(prompt.split(" "))
        # ensure Long dtype for embeddings and correct device
        return torch.tensor(ids, dtype=torch.long, device=self._device())

    # ----- public API -----
    def generate(self, prompt: str, max_new_tokens: int) -> List[str]:
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            idx = self._to_idx(prompt).unsqueeze(0)        # (1, T) on model device
            encoded = self.model.generate(idx, max_new_tokens)[0]  # (T+...)
            tokens = encoded.tolist()
        if was_training:
            self.model.train()
        return self.loader.decode(tokens)

    def generate_from_prompts(self, max_new_tokens: int, display: bool = True) -> List[List[str]]:
        outs: List[List[str]] = []
        for p in self.prompts:
            outs.append(self.generate(p, max_new_tokens))
        if display:
            print("Generated sample: ")
            for out in outs:
                print(f"\n {out} \n", flush=True)
        return outs
