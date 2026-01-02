from data.base_loader import BaseLoader, DataMode
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from collections import Counter
from typing import List, Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ShakespeareDataLoader(BaseLoader):
    def __init__(self, batch_size: int, block_size: int, train_frac: float = 0.9):
        self.batch_size = batch_size
        self.block_size = block_size
        self.train_frac = train_frac
        with open("data/input.txt", 'r', encoding='utf8') as f:
            self.text = f.read()
        c = Counter(self.text)
        print(f"Char freqs:\n {c}")

        self._itos = sorted(list(set(self.text)))  # index -> char
        print(f"Character set: {self._itos}")
        self._stoi = { ch:i for i, ch in enumerate(self._itos) }  # char -> index
        self._vocab_size = len(self._itos)
        self._create_train_test_splits()
        
    def encode(self, text: List[str]) -> List[int]:
        """Input text is list of chars."""
        return [self._stoi[ch] for ch in text]
    
    def decode(self, l: List[int]) -> str:
        return ''.join(self._itos[i] for i in l)

    def get_batch(self, mode: DataMode) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: document shapes of x, y
        """
        data = self._train_data if mode == DataMode.TRAIN else self._val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    def _create_train_test_splits(self):
        data = torch.tensor(self.encode(list(self.text)), dtype=torch.long)
        self.n = len(data)
        self.num_train = int(self.train_frac * self.n)
        self._train_data = data[:self.num_train]
        self._val_data = data[self.num_train:]
        print(f"total samples: {self.n}, train: {self.num_train}, val: {self.n - self.num_train}")
