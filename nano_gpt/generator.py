# nano_gpt/generator.py
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import List, Optional

from nano_gpt.data_loader import ShakespeareDataLoader
from nano_gpt.data_wt2_word import WT2WordDataLoader

class Generator:
    def __init__(
        self,
        model: nn.Module,
        loader: ShakespeareDataLoader | WT2WordDataLoader,
        char_level_tokenize: bool,
        sample_prompts: List[str],
    ):
        self.model = model
        self.loader = loader
        self.char_level_tokenize = char_level_tokenize
        self.prompts = sample_prompts

    
    # ----- public API -----
    def generate(self, prompt: str, max_new_tokens: int, greedy: bool = False) -> List[str]:
        """
        Generate continuation for a single prompt.
        If greedy=True, picks argmax at each step; otherwise samples from the distribution.
        """
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            idx = self._to_idx(prompt).unsqueeze(0)        # (1, T) on model device
            encoded = self.model.generate(
                idx, max_new_tokens, greedy=greedy
            )[0]  # (T+...)
            tokens = encoded.tolist()
        if was_training:
            self.model.train()
        return self.loader.decode(tokens)

    def generate_from_prompts(self, max_new_tokens: int, display: bool = True, greedy: bool = False) -> List[List[str]]:
        """
        Generate continuations for all configured sample prompts.
        Set greedy=True for argmax decoding.
        """
        outs: List[List[str]] = []
        for p in self.prompts:
            outs.append(self.generate(p, max_new_tokens, greedy=greedy))
        if display:
            print("Generated sample: ")
            for out in outs:
                print(f"\n {out} \n", flush=True)
        return outs

    # ----- internal helpers -----
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

