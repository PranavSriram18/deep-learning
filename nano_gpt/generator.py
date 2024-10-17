import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import Dict, List

from nano_gpt.bigram_model import BigramModel
from nano_gpt.data_loader import DataLoader

class Generator:
    # TODO - fix the type signature
    def __init__(
            self, 
            model: nn.Module, 
            loader: DataLoader, 
            char_level_chunk: bool,
            sample_prompts: List[str]):
        self.model = model
        self.loader = loader
        self.char_level_chunk = char_level_chunk
        self.prompts = sample_prompts

    def generate(self, prompt: str, max_new_tokens: int) -> List[str]:
        idx = self.to_idx(prompt)  # length T vector, where T is number of tokens in prompt
        idx = idx.unsqueeze(0)  # 1xT, since model expects a batch dimension
        encoded = self.model.generate(idx=idx, max_new_tokens=max_new_tokens)[0]
        return self.loader.decode(encoded.tolist())
    
    def generate_from_prompts(self, max_new_tokens: int, display: bool = True) -> List[List[str]]:
        ret = [self.generate(prompt, max_new_tokens) for prompt in self.prompts]
        if display:
            print("Generated sample: ")
            for out in ret:
                print(f"\n {out} \n", flush=True)
        return ret
        
    def to_idx(self, prompt: str) -> torch.Tensor:
        if self.char_level_chunk:
            return torch.tensor(self.loader.encode(list(prompt)))
        else:
            return torch.tensor(self.loader.encode(prompt.split(" ")))
