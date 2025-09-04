import torch  # type: ignore
import torch.nn as nn  # type: ignore
from collections import Counter
from typing import List, Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ShakespeareDataLoader:
    def __init__(self, batch_size: int, block_size: int):
        self.batch_size = batch_size
        self.block_size = block_size
        with open("nano_gpt/input.txt", 'r', encoding='utf8') as f:
            self.text = f.read()
        c = Counter(self.text)
        print(f"Char freqs:\n {c}")

        self.itos = sorted(list(set(self.text)))  # index -> char
        print(f"Character set: {self.itos}")
        self.stoi = { ch:i for i, ch in enumerate(self.itos) }  # char -> index
        self.vocab_size = len(self.itos)
        self._create_train_test_splits()
        
    def encode(self, text: List[str]) -> List[int]:
        """Input text is list of chars."""
        return [self.stoi[ch] for ch in text]
    
    def decode(self, l: List[int]) -> str:
        return ''.join(self.itos[i] for i in l)

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    def _create_train_test_splits(self):
        data = torch.tensor(self.encode(list(self.text)), dtype=torch.long)
        self.n = len(data)
        self.num_train = int(0.9 * self.n)
        self.train_data = data[:self.num_train]
        self.val_data = data[self.num_train:]
