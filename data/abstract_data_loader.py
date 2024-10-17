from abc import ABC, abstractmethod
from typing import List 

class AbstractDataLoader(ABC):
    @abstractmethod
    def encode(self, s: str) -> List[int]:
        """Encode text into a list of indices."""
        pass

    @abstractmethod
    def decode(self, l: list[int]) -> str:
        """Decode a list of indices into text."""
        pass 

    @abstractmethod
    def get_batch(self, split: str):
        """
        Get a batch of data.
        split: "train", "val", or "test"
        """
        pass

