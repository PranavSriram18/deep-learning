from abc import abstractmethod
from enum import Enum, auto 
from typing import Optional

import torch  # type: ignore

class DataMode(Enum):
    TRAIN = auto()
    EVAL = auto()
    TEST = auto()

class BaseLoader:
    def __init__(
        self, batch_size: int, vocab_size: Optional[int]
    ):
        self._batch_size = batch_size
        self._vocab_size = vocab_size

    def vocab_size(self) -> int:
        """Returns the effective vocab size for this data loader."""
        return self._vocab_size

    @abstractmethod
    def get_batch(self, mode: DataMode) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple x, y of data.
        Child classes should document shapes of x, y.
        """
        pass

    @abstractmethod
    def encode(self, tokens: list[str]) -> list[int]:
        """
        Encodes a list of tokens into a list of IDs.
        """
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> list[str]:
        """
        Decodes a list of IDs into a list of tokens.
        """
        pass
