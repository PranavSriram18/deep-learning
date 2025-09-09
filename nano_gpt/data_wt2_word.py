# nano_gpt/data_wt2_word.py
from collections import Counter
from typing import List, Tuple
import torch  # type: ignore
from datasets import load_dataset  # type: ignore

UNK_TOKEN = "<UNK>"

class WT2WordDataLoader:
    """
    Word-level WikiText-2 (raw) loader with vocab capping.
    - Vocabulary is built from the train split only.
    - Keeps the top `vocab_size - 1` words by frequency; all others map to <UNK>.
    - Returns next-token prediction batches: (B, T) -> (B, T).
    """

    def __init__(
        self,
        block_size: int,
        batch_size: int,
        vocab_size: int = 50_000,
        seed: int = 1337,
    ):
        assert vocab_size >= 1, "vocab_size must be >= 1"
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self._rng = torch.Generator().manual_seed(seed)

        # Load WikiText-2 raw (train/validation/test)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

        # Concatenate lines; keep newlines so paragraph boundaries remain
        train_text = "\n".join(ds["train"]["text"])
        val_text   = "\n".join(ds["validation"]["text"])

        # Simple whitespace tokenization (keeps punctuation attached to words)
        train_tokens = train_text.split()
        val_tokens   = val_text.split()

        # Build vocab from train only
        counter = Counter(train_tokens)
        # Reserve id 0 for <UNK>, then the top (vocab_size-1) tokens
        kept = [w for w, _ in counter.most_common(max(0, vocab_size - 1))]
        self.itos: List[str] = [UNK_TOKEN] + kept
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        def encode(tokens: List[str]) -> List[int]:
            unk = 0
            return [self.stoi.get(w, unk) for w in tokens]

        def decode(ids: List[int]) -> List[str]:
            n = len(self.itos)
            return [self.itos[i] if 0 <= i < n else UNK_TOKEN for i in ids]

        self.encode = encode
        self.decode = decode
        self.vocab_size_effective = len(self.itos)

        # Materialize token id streams
        self.train_ids = torch.tensor(encode(train_tokens), dtype=torch.long)
        self.val_ids   = torch.tensor(encode(val_tokens),   dtype=torch.long)

        # Cached lengths
        self.n_train = int(self.train_ids.numel())
        self.n_val   = int(self.val_ids.numel())

        print(f"Successfully built WT2WordDataLoader with vocab size {
            self.vocab_size_effective}, block size {self.block_size}, and batch size {
                self.batch_size}", flush=True)

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_ids if split == "train" else self.val_ids
        # Sample start positions; ensure room for block_size+1 tokens
        hi = data.numel() - self.block_size - 1
        idx = torch.randint(0, hi, (self.batch_size,), generator=self._rng)
        x = torch.stack([data[i : i + self.block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in idx])
        return x, y
